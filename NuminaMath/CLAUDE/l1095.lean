import Mathlib

namespace sum_10_is_35_l1095_109597

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  third_term : 2 * a 3 = 5
  sum_4_12 : a 4 + a 12 = 9

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1))

/-- The theorem to be proved -/
theorem sum_10_is_35 (seq : ArithmeticSequence) : sum_n seq 10 = 35 := by
  sorry

end sum_10_is_35_l1095_109597


namespace tangent_line_implies_a_minus_b_l1095_109527

noncomputable def f (a b x : ℝ) : ℝ := x + a / x + b

theorem tangent_line_implies_a_minus_b (a b : ℝ) :
  (∀ x ≠ 0, HasDerivAt (f a b) (1 - a / (x^2)) x) →
  (f a b 1 = 1 + a + b) →
  (HasDerivAt (f a b) (-2) 1) →
  (∃ c, ∀ x, f a b x = -2 * x + c) →
  a - b = 4 := by
sorry

end tangent_line_implies_a_minus_b_l1095_109527


namespace three_valid_configurations_l1095_109595

/-- Represents a square in the configuration --/
structure Square :=
  (label : Char)

/-- Represents the F-shaped configuration --/
def FConfiguration : Finset Square := sorry

/-- The set of additional lettered squares --/
def AdditionalSquares : Finset Square := sorry

/-- Predicate to check if a configuration is valid (foldable into a cube with one open non-bottom side) --/
def IsValidConfiguration (config : Finset Square) : Prop := sorry

/-- The number of valid configurations --/
def ValidConfigurationsCount : ℕ := sorry

/-- Theorem stating that there are exactly 3 valid configurations --/
theorem three_valid_configurations :
  ValidConfigurationsCount = 3 := by sorry

end three_valid_configurations_l1095_109595


namespace first_triangle_height_l1095_109572

/-- Given two triangles where the second has double the area of the first,
    prove that the height of the first triangle is 12 cm. -/
theorem first_triangle_height
  (base1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_base2 : base2 = 20)
  (h_height2 : height2 = 18)
  (h_area_relation : base2 * height2 = 2 * base1 * (12 : ℝ)) :
  ∃ (height1 : ℝ), height1 = 12 ∧ base1 * height1 = (1/2) * base2 * height2 :=
by sorry

end first_triangle_height_l1095_109572


namespace f_properties_l1095_109509

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then -x - 1
  else if -1 ≤ x ∧ x ≤ 1 then -x^2 + 1
  else x - 1

-- Theorem statement
theorem f_properties :
  (f 2 = 1 ∧ f (-2) = 1) ∧
  (∀ a : ℝ, f a = 1 ↔ a = -2 ∨ a = 0 ∨ a = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 ≤ x ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < y ∧ y ≤ -1) → f x > f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < y ∧ y ≤ 1) → f x > f y) :=
by sorry

end f_properties_l1095_109509


namespace algebraic_equality_l1095_109584

theorem algebraic_equality (a b : ℝ) : 
  (2*a^2 - 4*a*b + b^2 = -3*a^2 + 2*a*b - 5*b^2) → 
  (2*a^2 - 4*a*b + b^2 + 3*a^2 - 2*a*b + 5*b^2 = 5*a^2 + 6*b^2 - 6*a*b) := by
  sorry

end algebraic_equality_l1095_109584


namespace chinese_dinner_cost_l1095_109593

theorem chinese_dinner_cost (num_people : ℕ) (num_appetizers : ℕ) (appetizer_cost : ℚ)
  (tip_percentage : ℚ) (rush_fee : ℚ) (total_spent : ℚ) :
  num_people = 4 →
  num_appetizers = 2 →
  appetizer_cost = 6 →
  tip_percentage = 0.2 →
  rush_fee = 5 →
  total_spent = 77 →
  ∃ (main_meal_cost : ℚ),
    main_meal_cost * num_people +
    num_appetizers * appetizer_cost +
    (main_meal_cost * num_people + num_appetizers * appetizer_cost) * tip_percentage +
    rush_fee = total_spent ∧
    main_meal_cost = 12 :=
by sorry

end chinese_dinner_cost_l1095_109593


namespace oil_price_reduction_l1095_109528

/-- Proves that given a 20% reduction in the price of oil, if a housewife can obtain 10 kgs more for Rs. 1500 after the reduction, then the reduced price per kg is Rs. 30. -/
theorem oil_price_reduction (original_price : ℝ) : 
  (1500 / (0.8 * original_price) - 1500 / original_price = 10) → 
  (0.8 * original_price = 30) := by
sorry

end oil_price_reduction_l1095_109528


namespace students_per_class_l1095_109502

theorem students_per_class 
  (total_students : ℕ) 
  (num_classrooms : ℕ) 
  (h1 : total_students = 120) 
  (h2 : num_classrooms = 24) 
  (h3 : total_students % num_classrooms = 0) : 
  total_students / num_classrooms = 5 := by
sorry

end students_per_class_l1095_109502


namespace star_negative_two_five_l1095_109569

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- Theorem statement
theorem star_negative_two_five : star (-2) 5 = -273 := by
  sorry

end star_negative_two_five_l1095_109569


namespace axis_of_symmetry_point_relationship_t_range_max_t_value_l1095_109587

-- Define the parabola
def parabola (t x y : ℝ) : Prop := y = x^2 - 2*t*x + 1

-- Theorem for the axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ x y : ℝ, parabola t x y → parabola t (2*t - x) y := by sorry

-- Theorem for point relationship
theorem point_relationship (t m n : ℝ) :
  parabola t (t-2) m → parabola t (t+3) n → m < n := by sorry

-- Theorem for t range
theorem t_range (t : ℝ) :
  (∀ x₁ y₁ y₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ y₁ ∧ parabola t 3 y₂ ∧ y₁ ≤ y₂) 
  → t ≤ 1 := by sorry

-- Theorem for maximum t value
theorem max_t_value :
  ∃ t_max : ℝ, t_max = 5 ∧ 
  ∀ t y₁ y₂ : ℝ, parabola t (t+1) y₁ ∧ parabola t (2*t-4) y₂ ∧ y₁ ≥ y₂ 
  → t ≤ t_max := by sorry

end axis_of_symmetry_point_relationship_t_range_max_t_value_l1095_109587


namespace minimum_rental_fee_for_class_trip_l1095_109520

/-- Calculates the minimum rental fee for a class trip --/
def minimum_rental_fee (total_students : ℕ) 
  (small_boat_capacity small_boat_cost large_boat_capacity large_boat_cost : ℕ) : ℕ :=
  let large_boats := total_students / large_boat_capacity
  let remaining_students := total_students % large_boat_capacity
  let small_boats := (remaining_students + small_boat_capacity - 1) / small_boat_capacity
  large_boats * large_boat_cost + small_boats * small_boat_cost

theorem minimum_rental_fee_for_class_trip :
  minimum_rental_fee 48 3 16 5 24 = 232 :=
by sorry

end minimum_rental_fee_for_class_trip_l1095_109520


namespace min_value_xy_l1095_109512

theorem min_value_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 5/x + 3/y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 5/a + 3/b = 2 → x * y ≤ a * b :=
by sorry

end min_value_xy_l1095_109512


namespace max_value_interval_l1095_109575

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

theorem max_value_interval (a : ℝ) (h1 : a ≤ 4) :
  (∃ (x : ℝ), x ∈ Set.Ioo (3 - a^2) a ∧
   ∀ (y : ℝ), y ∈ Set.Ioo (3 - a^2) a → f y ≤ f x) →
  Real.sqrt 2 < a ∧ a ≤ 4 :=
by sorry

end max_value_interval_l1095_109575


namespace xiao_hong_pen_purchase_l1095_109521

theorem xiao_hong_pen_purchase (total_money : ℝ) (pen_cost : ℝ) (notebook_cost : ℝ) 
  (notebooks_bought : ℕ) (h1 : total_money = 18) (h2 : pen_cost = 3) 
  (h3 : notebook_cost = 3.6) (h4 : notebooks_bought = 2) :
  ∃ (pens : ℕ), pens ∈ ({1, 2, 3} : Set ℕ) ∧ 
  (notebooks_bought : ℝ) * notebook_cost + (pens : ℝ) * pen_cost ≤ total_money :=
sorry

end xiao_hong_pen_purchase_l1095_109521


namespace shifted_quadratic_sum_of_coefficients_l1095_109505

-- Define the original quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 3)

-- Theorem statement
theorem shifted_quadratic_sum_of_coefficients :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 51) := by
sorry

end shifted_quadratic_sum_of_coefficients_l1095_109505


namespace functional_equation_characterization_l1095_109566

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

/-- The characterization of functions satisfying the functional equation -/
theorem functional_equation_characterization (f : ℕ → ℕ) 
  (h : FunctionalEquation f) : 
  ∃ d : ℕ, d > 0 ∧ ∀ m : ℕ, ∃ k : ℕ, f m = k * d :=
by sorry

end functional_equation_characterization_l1095_109566


namespace binomial_coefficient_two_l1095_109583

theorem binomial_coefficient_two (n : ℕ+) : (n.val.choose 2) = n.val * (n.val - 1) / 2 := by
  sorry

end binomial_coefficient_two_l1095_109583


namespace sin_double_angle_with_tan_l1095_109596

theorem sin_double_angle_with_tan (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := by sorry

end sin_double_angle_with_tan_l1095_109596


namespace gwen_gave_away_seven_games_l1095_109581

/-- The number of games Gwen gave away -/
def games_given_away (initial_games : ℕ) (remaining_games : ℕ) : ℕ :=
  initial_games - remaining_games

/-- Proof that Gwen gave away 7 games -/
theorem gwen_gave_away_seven_games :
  let initial_games : ℕ := 98
  let remaining_games : ℕ := 91
  games_given_away initial_games remaining_games = 7 := by
  sorry

end gwen_gave_away_seven_games_l1095_109581


namespace train_speed_proof_l1095_109508

/-- The average speed of a train with stoppages, in km/h -/
def speed_with_stoppages : ℝ := 60

/-- The duration of stoppages per hour, in minutes -/
def stoppage_duration : ℝ := 15

/-- The average speed of a train without stoppages, in km/h -/
def speed_without_stoppages : ℝ := 80

theorem train_speed_proof :
  speed_without_stoppages * ((60 - stoppage_duration) / 60) = speed_with_stoppages :=
by sorry

end train_speed_proof_l1095_109508


namespace semicircle_perimeter_approx_l1095_109514

/-- The perimeter of a semicircle with radius 12 is approximately 61.7 -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  abs ((12 * Real.pi + 24) - 61.7) < ε :=
by sorry

end semicircle_perimeter_approx_l1095_109514


namespace volume_cube_inscribed_sphere_l1095_109522

/-- The volume of a cube inscribed in a sphere of radius R -/
theorem volume_cube_inscribed_sphere (R : ℝ) (R_pos : 0 < R) :
  ∃ (V : ℝ), V = (8 / 9) * Real.sqrt 3 * R^3 :=
sorry

end volume_cube_inscribed_sphere_l1095_109522


namespace f_properties_l1095_109504

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos φ + 2 * Real.cos (ω * x) * Real.sin φ

theorem f_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ∃ φ',
    (∀ x, f ω φ x = 2 * Real.sin (2 * x + φ')) ∧
    (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≤ 2) ∧
    (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≥ 0) ∧
    (∃ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x = 2) ∧
    ((∃ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x = 0) ∨
     (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≥ 1)) := by
  sorry

end f_properties_l1095_109504


namespace total_gift_cost_l1095_109535

def engagement_ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def diamond_bracelet_cost : ℕ := 2 * engagement_ring_cost

theorem total_gift_cost : engagement_ring_cost + car_cost + diamond_bracelet_cost = 14000 := by
  sorry

end total_gift_cost_l1095_109535


namespace intersection_implies_union_l1095_109550

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3) = 0}
def N : Set ℝ := {x | (x - 4) * (x - 1) = 0}

-- State the theorem
theorem intersection_implies_union (a : ℝ) : 
  (M a ∩ N ≠ ∅) → (M a ∪ N = {1, 3, 4}) := by
  sorry

end intersection_implies_union_l1095_109550


namespace right_triangle_circle_properties_l1095_109576

/-- Properties of right triangles relating inscribed and circumscribed circles -/
theorem right_triangle_circle_properties (a b c r R p : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → p > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  p = a + b + c →    -- Perimeter
  r = (a + b - c) / 2 →  -- Inradius formula
  R = c / 2 →        -- Circumradius formula
  (p / c - r / R = 2) ∧
  (r / R ≤ 1 / (Real.sqrt 2 + 1)) ∧
  (r / R = 1 / (Real.sqrt 2 + 1) ↔ a = b) :=
by sorry


end right_triangle_circle_properties_l1095_109576


namespace point_movement_l1095_109513

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given conditions and proof goal -/
theorem point_movement :
  let A : Point := ⟨a - 5, 2 * b - 1⟩
  let B : Point := ⟨3 * a + 2, b + 3⟩
  let C : Point := ⟨a, b⟩
  A.x = 0 →  -- A lies on y-axis
  B.y = 0 →  -- B lies on x-axis
  (⟨C.x + 2, C.y - 3⟩ : Point) = ⟨7, -6⟩ := by
  sorry

end point_movement_l1095_109513


namespace equation_solutions_l1095_109503

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 9 = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, (-x)^3 = (-8)^2 ↔ x = -4) := by
  sorry

end equation_solutions_l1095_109503


namespace first_cat_blue_eyed_count_l1095_109551

/-- The number of blue-eyed kittens in the first cat's litter -/
def blue_eyed_first_cat : ℕ := sorry

/-- The number of brown-eyed kittens in the first cat's litter -/
def brown_eyed_first_cat : ℕ := 7

/-- The number of blue-eyed kittens in the second cat's litter -/
def blue_eyed_second_cat : ℕ := 4

/-- The number of brown-eyed kittens in the second cat's litter -/
def brown_eyed_second_cat : ℕ := 6

/-- The percentage of blue-eyed kittens among all kittens -/
def blue_eyed_percentage : ℚ := 35 / 100

theorem first_cat_blue_eyed_count :
  blue_eyed_first_cat = 3 :=
by
  sorry

end first_cat_blue_eyed_count_l1095_109551


namespace sum_of_x_and_y_l1095_109567

theorem sum_of_x_and_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 + y^2 = 1) (h4 : (3*x - 4*x^3) * (3*y - 4*y^3) = -1/2) : 
  x + y = Real.sqrt 6 / 2 := by
sorry

end sum_of_x_and_y_l1095_109567


namespace quadratic_inequality_necessary_condition_l1095_109532

theorem quadratic_inequality_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) →
  (0 ≤ a ∧ a ≤ 4) :=
by sorry

end quadratic_inequality_necessary_condition_l1095_109532


namespace f_g_product_positive_l1095_109553

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x
def monotone_increasing_on (g : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → g x ≤ g y

-- State the theorem
theorem f_g_product_positive
  (h_f_odd : is_odd f)
  (h_f_decr : monotone_decreasing_on f {x | x < 0})
  (h_g_even : is_even g)
  (h_g_incr : monotone_increasing_on g {x | x ≤ 0})
  (h_f_1 : f 1 = 0)
  (h_g_1 : g 1 = 0) :
  {x : ℝ | f x * g x > 0} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x > 1} :=
sorry

end f_g_product_positive_l1095_109553


namespace lunchroom_tables_l1095_109559

theorem lunchroom_tables (students_per_table : ℕ) (total_students : ℕ) (h1 : students_per_table = 6) (h2 : total_students = 204) :
  total_students / students_per_table = 34 := by
  sorry

end lunchroom_tables_l1095_109559


namespace ant_beetle_distance_difference_l1095_109545

/-- Calculates the percentage difference in distance traveled between an ant and a beetle -/
theorem ant_beetle_distance_difference :
  let ant_distance : ℝ := 600  -- meters
  let ant_time : ℝ := 12       -- minutes
  let beetle_speed : ℝ := 2.55 -- km/h
  
  let ant_speed : ℝ := (ant_distance / 1000) / (ant_time / 60)
  let beetle_distance : ℝ := (beetle_speed * ant_time) / 60 * 1000
  
  let difference : ℝ := ant_distance - beetle_distance
  let percentage_difference : ℝ := (difference / ant_distance) * 100
  
  percentage_difference = 15 := by
  sorry

end ant_beetle_distance_difference_l1095_109545


namespace donut_selection_equals_object_distribution_l1095_109501

/-- The number of ways to select n donuts from k types with at least one of a specific type -/
def donut_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 2) (k - 1)

/-- The number of ways to distribute m objects into k distinct boxes -/
def object_distribution (m k : ℕ) : ℕ :=
  Nat.choose (m + k - 1) (k - 1)

theorem donut_selection_equals_object_distribution :
  donut_selections 5 4 = object_distribution 4 4 :=
by sorry

end donut_selection_equals_object_distribution_l1095_109501


namespace square_sum_of_difference_and_product_l1095_109582

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 10) : 
  a^2 + b^2 = 29 := by
sorry

end square_sum_of_difference_and_product_l1095_109582


namespace right_triangle_third_side_l1095_109548

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 3 ∧ b = 5 ∧ (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2) → c = Real.sqrt 34 ∨ c = 4 := by
  sorry

end right_triangle_third_side_l1095_109548


namespace min_value_reciprocal_sum_l1095_109540

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 2) :
  (1 / a + 1 / b) ≥ (5 + 2 * Real.sqrt 6) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = (5 + 2 * Real.sqrt 6) / 2 :=
by sorry

end min_value_reciprocal_sum_l1095_109540


namespace percentage_sum_l1095_109542

theorem percentage_sum (A B C : ℝ) : 
  (0.45 * A = 270) → 
  (0.35 * B = 210) → 
  (0.25 * C = 150) → 
  (0.75 * A + 0.65 * B + 0.45 * C = 1110) := by
sorry

end percentage_sum_l1095_109542


namespace expression_equals_seventeen_l1095_109556

theorem expression_equals_seventeen : 3 + 4 * 5 - 6 = 17 := by
  sorry

end expression_equals_seventeen_l1095_109556


namespace total_bills_is_30_l1095_109564

/-- Represents the number of $10 bills -/
def num_ten_bills : ℕ := 27

/-- Represents the number of $20 bills -/
def num_twenty_bills : ℕ := 3

/-- Represents the total value of all bills in dollars -/
def total_value : ℕ := 330

/-- Theorem stating that the total number of bills is 30 -/
theorem total_bills_is_30 : num_ten_bills + num_twenty_bills = 30 := by
  sorry

end total_bills_is_30_l1095_109564


namespace min_chopsticks_for_different_colors_l1095_109541

/-- Represents the number of pairs of chopsticks for each color -/
def pairs_per_color : ℕ := 4

/-- Represents the total number of colors -/
def total_colors : ℕ := 3

/-- Represents the total number of chopsticks -/
def total_chopsticks : ℕ := pairs_per_color * total_colors * 2

/-- 
Theorem: Given 12 pairs of chopsticks in 3 different colors (4 pairs each), 
the minimum number of chopsticks that must be taken out to guarantee 
two pairs of different colors is 11.
-/
theorem min_chopsticks_for_different_colors : ℕ := by
  sorry

end min_chopsticks_for_different_colors_l1095_109541


namespace two_vans_needed_l1095_109571

/-- The number of vans needed for a field trip -/
def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

/-- Proof that 2 vans are needed for the field trip -/
theorem two_vans_needed : vans_needed 4 2 6 = 2 := by
  sorry

end two_vans_needed_l1095_109571


namespace area_is_60_perimeter_is_40_l1095_109573

/-- Triangle with side lengths 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The area of the right triangle is 60 -/
theorem area_is_60 (t : RightTriangle) : (1/2) * t.a * t.b = 60 := by sorry

/-- The perimeter of the right triangle is 40 -/
theorem perimeter_is_40 (t : RightTriangle) : t.a + t.b + t.c = 40 := by sorry

end area_is_60_perimeter_is_40_l1095_109573


namespace probability_theorem_l1095_109530

/-- A regular hexagon --/
structure RegularHexagon where
  /-- The set of all sides and diagonals --/
  S : Finset ℝ
  /-- Number of sides --/
  num_sides : ℕ
  /-- Number of shorter diagonals --/
  num_shorter_diagonals : ℕ
  /-- Number of longer diagonals --/
  num_longer_diagonals : ℕ
  /-- Total number of segments --/
  total_segments : ℕ
  /-- Condition: num_sides = 6 --/
  sides_eq_six : num_sides = 6
  /-- Condition: num_shorter_diagonals = 6 --/
  shorter_diagonals_eq_six : num_shorter_diagonals = 6
  /-- Condition: num_longer_diagonals = 3 --/
  longer_diagonals_eq_three : num_longer_diagonals = 3
  /-- Condition: total_segments = num_sides + num_shorter_diagonals + num_longer_diagonals --/
  total_segments_eq_sum : total_segments = num_sides + num_shorter_diagonals + num_longer_diagonals

/-- The probability of selecting two segments of the same length --/
def probability_same_length (h : RegularHexagon) : ℚ :=
  33 / 105

/-- Theorem: The probability of selecting two segments of the same length is 33/105 --/
theorem probability_theorem (h : RegularHexagon) : 
  probability_same_length h = 33 / 105 := by
  sorry

end probability_theorem_l1095_109530


namespace max_value_theorem_l1095_109578

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  2 * x + y ≤ 46 / 11 := by
  sorry

end max_value_theorem_l1095_109578


namespace quadratic_inequality_transformation_l1095_109570

theorem quadratic_inequality_transformation (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c < 0 ↔ x < -2 ∨ x > -1/2) →
  (∀ x : ℝ, c*x^2 - b*x + a > 0 ↔ 1/2 < x ∧ x < 2) :=
by sorry

end quadratic_inequality_transformation_l1095_109570


namespace quadratic_equation_solution_l1095_109507

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 + 6*x + 8 = -(x + 2)*(x + 6) ↔ x = -2 ∨ x = -5 := by
sorry

end quadratic_equation_solution_l1095_109507


namespace in_class_calculation_l1095_109560

theorem in_class_calculation :
  (((4.2 : ℝ) + 2.2) / 0.08 = 80) ∧
  (100 / 0.4 / 2.5 = 100) := by
  sorry

end in_class_calculation_l1095_109560


namespace equation_solution_l1095_109546

theorem equation_solution (x : ℕ+) : (x.val - 1) * x.val * (4 * x.val + 1) = 750 ↔ x = 6 := by
  sorry

end equation_solution_l1095_109546


namespace second_polygon_sides_l1095_109586

/-- Given two regular polygons with the same perimeter, where the first has 38 sides
    and a side length twice that of the second, prove the second has 76 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) :
  s > 0 →
  38 * (2 * s) = n * s →
  n = 76 := by
  sorry

end second_polygon_sides_l1095_109586


namespace product_perfect_square_l1095_109598

theorem product_perfect_square (nums : Finset ℕ) : 
  (nums.card = 17) →
  (∀ n ∈ nums, ∃ (a b c d : ℕ), n = 2^a * 3^b * 5^c * 7^d) →
  ∃ (n1 n2 : ℕ), n1 ∈ nums ∧ n2 ∈ nums ∧ n1 ≠ n2 ∧ ∃ (m : ℕ), n1 * n2 = m^2 :=
by sorry

end product_perfect_square_l1095_109598


namespace alcohol_concentration_correct_l1095_109534

/-- The concentration of alcohol in the container after n operations --/
def alcohol_concentration (n : ℕ) : ℚ :=
  (12 - 9 * (3/4)^(n-1)) / (32 - 9 * (3/4)^(n-1))

/-- The amount of water in the container after n operations --/
def water_amount (n : ℕ) : ℚ :=
  20/3 * (2/3)^(n-1)

/-- The amount of alcohol in the container after n operations --/
def alcohol_amount (n : ℕ) : ℚ :=
  4 * (2/3)^(n-1) - 6 * (1/2)^n

/-- The theorem stating that the alcohol_concentration function correctly calculates
    the concentration of alcohol in the container after n operations --/
theorem alcohol_concentration_correct (n : ℕ) :
  alcohol_concentration n = alcohol_amount n / (water_amount n + alcohol_amount n) :=
by sorry

/-- The initial amount of water in the container --/
def initial_water : ℚ := 10

/-- The amount of alcohol added in the first step --/
def first_alcohol_addition : ℚ := 1

/-- The amount of alcohol added in the second step --/
def second_alcohol_addition : ℚ := 1/2

/-- The fraction of liquid removed in each step --/
def removal_fraction : ℚ := 1/3

/-- The ratio of alcohol added in each step compared to the previous step --/
def alcohol_addition_ratio : ℚ := 1/2

end alcohol_concentration_correct_l1095_109534


namespace quadratic_equation_solution_l1095_109526

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : (2*d)^2 + c*(2*d) + d = 0) : 
  c = (1 : ℝ) / 2 ∧ d = -(1 : ℝ) / 2 := by
  sorry

end quadratic_equation_solution_l1095_109526


namespace det_roots_cubic_l1095_109561

theorem det_roots_cubic (p q r a b c : ℝ) : 
  (a^3 - p*a^2 + q*a - r = 0) →
  (b^3 - p*b^2 + q*b - r = 0) →
  (c^3 - p*c^2 + q*c - r = 0) →
  let matrix := !![2 + a, 1, 1; 1, 2 + b, 1; 1, 1, 2 + c]
  Matrix.det matrix = r + 2*q + 4*p + 4 := by
  sorry

end det_roots_cubic_l1095_109561


namespace art_gallery_pieces_l1095_109525

theorem art_gallery_pieces (total : ℕ) 
  (displayed : ℕ) (sculptures_displayed : ℕ) 
  (paintings_not_displayed : ℕ) (sculptures_not_displayed : ℕ) :
  displayed = total / 3 →
  sculptures_displayed = displayed / 6 →
  paintings_not_displayed = (total - displayed) / 3 →
  sculptures_not_displayed = 1400 →
  total = 3150 :=
by
  sorry

end art_gallery_pieces_l1095_109525


namespace intersection_of_A_and_B_l1095_109529

def set_A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end intersection_of_A_and_B_l1095_109529


namespace point_P_coordinates_and_PQ_length_l1095_109523

def point_P (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3*n)

def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

def point_Q (n : ℝ) : ℝ × ℝ := (n, -4)

def parallel_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

theorem point_P_coordinates_and_PQ_length :
  ∃ n : ℝ,
    let p := point_P n
    let q := point_Q n
    fourth_quadrant p ∧
    distance_to_x_axis p = distance_to_y_axis p + 1 ∧
    parallel_to_x_axis p q ∧
    p = (6, -7) ∧
    |p.1 - q.1| = 3 :=
by sorry

end point_P_coordinates_and_PQ_length_l1095_109523


namespace min_value_theorem_l1095_109506

theorem min_value_theorem (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_eq_one : x + y + z + w = 1) :
  (x + y + z) / (x * y * z * w) ≥ 144 := by
  sorry

end min_value_theorem_l1095_109506


namespace circle_P_equation_l1095_109519

/-- The curve C defined by the distance ratio condition -/
def C (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2 = 1)

/-- The line l intersecting curve C -/
def l (x y : ℝ) (k : ℝ) : Prop :=
  (y = k * (x - 1) - 1)

/-- Points A and B are on both C and l -/
def A_and_B_on_C_and_l (x₁ y₁ x₂ y₂ k : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ k ∧ l x₂ y₂ k

/-- AB is the diameter of circle P centered at (1, -1) -/
def P_diameter (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = -1

theorem circle_P_equation (x₁ y₁ x₂ y₂ k : ℝ) :
  A_and_B_on_C_and_l x₁ y₁ x₂ y₂ k →
  P_diameter x₁ y₁ x₂ y₂ →
  k = 2/3 →
  ∀ x y, (x - 1)^2 + (y + 1)^2 = 13/30 :=
sorry

end circle_P_equation_l1095_109519


namespace valid_arrangements_l1095_109517

/- Define the number of students and schools -/
def total_students : ℕ := 4
def num_schools : ℕ := 2
def students_per_school : ℕ := 2

/- Define a function to calculate the number of arrangements -/
def num_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  if n = total_students ∧ k = num_schools ∧ m = students_per_school ∧ n = k * m then
    2 * (Nat.factorial m) * (Nat.factorial m)
  else
    0

/- Theorem statement -/
theorem valid_arrangements :
  num_arrangements total_students num_schools students_per_school = 8 :=
by sorry

end valid_arrangements_l1095_109517


namespace team_supporters_equal_positive_responses_l1095_109544

-- Define the four teams
inductive Team
| Spartak
| Dynamo
| Zenit
| Lokomotiv

-- Define the result of a match
inductive MatchResult
| Win
| Lose

-- Define a function to represent fan behavior
def fanResponse (team : Team) (result : MatchResult) : Bool :=
  match result with
  | MatchResult.Win => true
  | MatchResult.Lose => false

-- Define the theorem
theorem team_supporters_equal_positive_responses 
  (match1 : Team → MatchResult) 
  (match2 : Team → MatchResult)
  (positiveResponses : Team → Nat)
  (h1 : ∀ t, (match1 t = MatchResult.Win) ≠ (match2 t = MatchResult.Win))
  (h2 : positiveResponses Team.Spartak = 200)
  (h3 : positiveResponses Team.Dynamo = 300)
  (h4 : positiveResponses Team.Zenit = 500)
  (h5 : positiveResponses Team.Lokomotiv = 600)
  : ∀ t, positiveResponses t = 
    (if fanResponse t (match1 t) then 1 else 0) + 
    (if fanResponse t (match2 t) then 1 else 0) := by
  sorry


end team_supporters_equal_positive_responses_l1095_109544


namespace polynomial_evaluation_l1095_109579

/-- Given real numbers a, b, and c, and polynomials g and f as defined,
    prove that f(-1) = -29041 -/
theorem polynomial_evaluation (a b c : ℝ) : 
  let g := fun (x : ℝ) => x^3 + a*x^2 + x + 20
  let f := fun (x : ℝ) => x^4 + x^3 + b*x^2 + 200*x + c
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  f (-1) = -29041 := by
sorry

end polynomial_evaluation_l1095_109579


namespace johns_country_club_payment_l1095_109599

/-- Represents the cost John pays for the country club membership in the first year -/
def johns_payment (num_members : ℕ) (joining_fee_pp : ℕ) (monthly_cost_pp : ℕ) : ℕ :=
  let total_joining_fee := num_members * joining_fee_pp
  let total_monthly_cost := num_members * monthly_cost_pp * 12
  let total_cost := total_joining_fee + total_monthly_cost
  total_cost / 2

/-- Proves that John's payment for the first year is $32000 given the problem conditions -/
theorem johns_country_club_payment :
  johns_payment 4 4000 1000 = 32000 := by
sorry

end johns_country_club_payment_l1095_109599


namespace least_n_for_fraction_inequality_l1095_109516

theorem least_n_for_fraction_inequality : 
  (∃ n : ℕ, n > 0 ∧ (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < 4 → (1 : ℚ) / m - (1 : ℚ) / (m + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15) :=
by sorry

end least_n_for_fraction_inequality_l1095_109516


namespace find_x_l1095_109549

theorem find_x : ∃ x : ℚ, (3 * x + 4) / 6 = 15 ∧ x = 86 / 3 := by
  sorry

end find_x_l1095_109549


namespace find_number_l1095_109591

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 5) = 105 := by
  sorry

end find_number_l1095_109591


namespace total_cost_is_119_l1095_109536

-- Define the number of games and ticket prices for each month
def this_month_games : ℕ := 9
def this_month_price : ℕ := 5
def last_month_games : ℕ := 8
def last_month_price : ℕ := 4
def next_month_games : ℕ := 7
def next_month_price : ℕ := 6

-- Define the total cost function
def total_cost : ℕ :=
  this_month_games * this_month_price +
  last_month_games * last_month_price +
  next_month_games * next_month_price

-- Theorem statement
theorem total_cost_is_119 : total_cost = 119 := by
  sorry

end total_cost_is_119_l1095_109536


namespace cube_surface_area_l1095_109537

/-- The surface area of a cube with edge length 8 cm is 384 cm². -/
theorem cube_surface_area : 
  let edge_length : ℝ := 8
  let surface_area : ℝ := 6 * edge_length * edge_length
  surface_area = 384 :=
by sorry

end cube_surface_area_l1095_109537


namespace alok_rice_order_l1095_109580

def chapatis : ℕ := 16
def mixed_vegetable : ℕ := 7
def ice_cream_cups : ℕ := 6
def cost_chapati : ℕ := 6
def cost_rice : ℕ := 45
def cost_mixed_vegetable : ℕ := 70
def cost_ice_cream : ℕ := 40
def total_paid : ℕ := 1051

theorem alok_rice_order :
  ∃ (rice_plates : ℕ),
    rice_plates = 5 ∧
    total_paid = chapatis * cost_chapati +
                 rice_plates * cost_rice +
                 mixed_vegetable * cost_mixed_vegetable +
                 ice_cream_cups * cost_ice_cream :=
by sorry

end alok_rice_order_l1095_109580


namespace calculate_expression_l1095_109531

theorem calculate_expression : 3 * 7.5 * (6 + 4) / 2.5 = 90 := by sorry

end calculate_expression_l1095_109531


namespace sum_of_coefficients_l1095_109589

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 - 3*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -33 := by
  sorry

end sum_of_coefficients_l1095_109589


namespace jack_and_beanstalk_height_l1095_109594

/-- The height of the sky island in Jack and the Beanstalk --/
def sky_island_height (day_climb : ℕ) (night_slide : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - 1) * (day_climb - night_slide) + day_climb

theorem jack_and_beanstalk_height :
  sky_island_height 25 3 64 = 1411 := by
  sorry

end jack_and_beanstalk_height_l1095_109594


namespace lemonade_syrup_parts_l1095_109592

/-- Given a solution with 8 parts water for every L parts lemonade syrup,
    prove that if removing 2.1428571428571423 parts and replacing with water
    results in 25% lemonade syrup, then L = 2.6666666666666665 -/
theorem lemonade_syrup_parts (L : ℝ) : 
  (L = 0.25 * (8 + L)) → L = 2.6666666666666665 := by
  sorry

end lemonade_syrup_parts_l1095_109592


namespace proportionality_problem_l1095_109539

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^2,
    prove that x = 1/16 when z = 32, given that x = 4 when z = 8. -/
theorem proportionality_problem (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h₁ : x = k₁ * y^4)
    (h₂ : y * z^2 = k₂)
    (h₃ : x = 4 ∧ z = 8 → k₁ * k₂^4 = 67108864) :
    z = 32 → x = 1/16 := by
  sorry

end proportionality_problem_l1095_109539


namespace turtle_difference_l1095_109555

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The total number of turtles Marion and Martha received together -/
def total_turtles : ℕ := 100

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := total_turtles - martha_turtles

/-- Marion received more turtles than Martha -/
axiom marion_more : marion_turtles > martha_turtles

theorem turtle_difference : marion_turtles - martha_turtles = 20 := by
  sorry

end turtle_difference_l1095_109555


namespace jordan_rectangle_width_l1095_109538

/-- Given two rectangles of equal area, where one rectangle measures 4.5 inches by 19.25 inches,
    and the other rectangle is 3.75 inches long, the width of the second rectangle is 23.1 inches. -/
theorem jordan_rectangle_width (carol_length carol_width jordan_length : ℝ)
  (h1 : carol_length = 4.5)
  (h2 : carol_width = 19.25)
  (h3 : jordan_length = 3.75)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 23.1 :=
by sorry


end jordan_rectangle_width_l1095_109538


namespace fraction_value_l1095_109511

/-- Given that x is four times y, y is three times z, and z is five times w,
    prove that (x * z) / (y * w) = 20 -/
theorem fraction_value (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  (x * z) / (y * w) = 20 := by
  sorry

end fraction_value_l1095_109511


namespace stationery_box_sheet_count_l1095_109563

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents the usage of a stationery box -/
structure Usage where
  sheetsPerLetter : ℕ
  usedAllEnvelopes : Bool
  usedAllSheets : Bool
  leftoverSheets : ℕ
  leftoverEnvelopes : ℕ

theorem stationery_box_sheet_count (box : StationeryBox) 
  (ann_usage : Usage) (bob_usage : Usage) :
  ann_usage.sheetsPerLetter = 2 →
  bob_usage.sheetsPerLetter = 4 →
  ann_usage.usedAllEnvelopes = true →
  ann_usage.leftoverSheets = 30 →
  bob_usage.usedAllSheets = true →
  bob_usage.leftoverEnvelopes = 20 →
  box.sheets = 40 := by
sorry

end stationery_box_sheet_count_l1095_109563


namespace P_on_xoz_plane_l1095_109590

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- The given point P -/
def P : Point3D := ⟨-2, 0, 3⟩

/-- Theorem: Point P lies on the xoz plane -/
theorem P_on_xoz_plane : P ∈ xoz_plane := by
  sorry


end P_on_xoz_plane_l1095_109590


namespace tangent_line_at_one_f_lower_bound_l1095_109554

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (x - a) - Real.log x - Real.log a

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.exp (x - a) - 1 / x

theorem tangent_line_at_one (a : ℝ) (ha : a > 0) :
  f_prime a 1 = 1 → ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ ∀ x : ℝ, f a x = m * x + b := by sorry

theorem f_lower_bound (a : ℝ) (ha : 0 < a) (ha2 : a < (Real.sqrt 5 - 1) / 2) :
  ∀ x : ℝ, x > 0 → f a x > a / (a + 1) := by sorry

end tangent_line_at_one_f_lower_bound_l1095_109554


namespace inverse_function_property_l1095_109585

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property that f and f_inv are inverse functions
def are_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the property that f(x+2) and f_inv(x-1) are inverse functions
def special_inverse_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f ((f_inv (x - 1)) + 2) = x ∧ f_inv (f (x + 2) - 1) = x

-- State the theorem
theorem inverse_function_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : are_inverse f f_inv) 
  (h2 : special_inverse_property f f_inv) : 
  f_inv 2007 - f_inv 1 = 4012 := by
sorry

end inverse_function_property_l1095_109585


namespace parking_lot_wheels_l1095_109552

/-- The number of wheels on a car -/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a bike -/
def wheels_per_bike : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 2

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * wheels_per_car + num_bikes * wheels_per_bike

theorem parking_lot_wheels : total_wheels = 44 := by
  sorry

end parking_lot_wheels_l1095_109552


namespace product_of_numbers_with_given_sum_and_difference_l1095_109574

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 40 ∧ x - y = 10 → x * y = 375 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l1095_109574


namespace shelf_position_l1095_109562

theorem shelf_position (wall_width : ℝ) (picture_width : ℝ) 
  (hw : wall_width = 26)
  (hp : picture_width = 4) :
  let picture_center := wall_width / 2
  let shelf_left_edge := picture_center + picture_width / 2
  shelf_left_edge = 15 := by
  sorry

end shelf_position_l1095_109562


namespace rectangle_rotation_forms_cylinder_l1095_109568

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_positive : width > 0
  height_positive : height > 0

/-- Represents the solid formed by rotating a rectangle -/
inductive RotatedSolid
  | Cylinder
  | Other

/-- Function that determines the shape of the solid formed by rotating a rectangle -/
def rotate_rectangle (rect : Rectangle) : RotatedSolid := sorry

/-- Theorem stating that rotating a rectangle forms a cylinder -/
theorem rectangle_rotation_forms_cylinder (rect : Rectangle) :
  rotate_rectangle rect = RotatedSolid.Cylinder := by sorry

end rectangle_rotation_forms_cylinder_l1095_109568


namespace quadratic_function_properties_l1095_109577

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x + b

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b : ℝ), f a b (-1) = 0 →
  (b = 2*a - 1) ∧
  (a = -1 → ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f (-1) (-3) x ≤ f (-1) (-3) y) :=
by sorry

end quadratic_function_properties_l1095_109577


namespace certain_number_exists_l1095_109533

theorem certain_number_exists : ∃ x : ℝ, (1.78 * x) / 5.96 = 377.8020134228188 := by
  sorry

end certain_number_exists_l1095_109533


namespace perpendicular_bisector_equation_l1095_109558

/-- The perpendicular bisector of a line segment with endpoints (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem perpendicular_bisector_equation :
  perpendicular_bisector 1 3 5 (-1) = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} := by
  sorry

end perpendicular_bisector_equation_l1095_109558


namespace increase_by_percentage_l1095_109500

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 450 → percentage = 75 → result = initial * (1 + percentage / 100) → result = 787.5 := by
  sorry

end increase_by_percentage_l1095_109500


namespace tangent_point_and_zeros_l1095_109510

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.exp x + 2 * a * x - x + 3 - a^2

theorem tangent_point_and_zeros (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ (∀ y : ℝ, f a y ≥ 0)) ↔ a = Real.log 3 - 3 ∧
  (∀ x : ℝ, x > 0 →
    (((a ≤ -Real.sqrt 5 ∨ a = Real.log 3 - 3 ∨ a > Real.sqrt 5) →
      (∃! y : ℝ, y > 0 ∧ f a y = 0)) ∧
    ((-Real.sqrt 5 < a ∧ a < Real.log 3 - 3) →
      (∃ y z : ℝ, 0 < y ∧ y < z ∧ f a y = 0 ∧ f a z = 0 ∧
        ∀ w : ℝ, 0 < w ∧ w ≠ y ∧ w ≠ z → f a w ≠ 0)) ∧
    ((Real.log 3 - 3 < a ∧ a ≤ Real.sqrt 5) →
      (∀ y : ℝ, y > 0 → f a y ≠ 0)))) :=
sorry

end tangent_point_and_zeros_l1095_109510


namespace metaphase_mitosis_observable_l1095_109543

/-- Represents the types of cell division that can occur in testis --/
inductive CellDivisionType
| Mitosis
| Meiosis

/-- Represents the phases of mitosis --/
inductive MitosisPhase
| Prophase
| Metaphase
| Anaphase
| Telophase

/-- Represents a cell in a testis slice --/
structure TestisCell where
  divisionType : CellDivisionType
  phase : Option MitosisPhase

/-- Represents a locust testis slice --/
structure LocustTestisSlice where
  cells : List TestisCell

/-- Condition: Both meiosis and mitosis can occur in the testis --/
def testisCanUndergoMitosisAndMeiosis (slice : LocustTestisSlice) : Prop :=
  ∃ (c1 c2 : TestisCell), c1 ∈ slice.cells ∧ c2 ∈ slice.cells ∧
    c1.divisionType = CellDivisionType.Mitosis ∧
    c2.divisionType = CellDivisionType.Meiosis

/-- Theorem: In locust testis slices, cells in the metaphase of mitosis can be observed --/
theorem metaphase_mitosis_observable (slice : LocustTestisSlice) 
  (h : testisCanUndergoMitosisAndMeiosis slice) :
  ∃ (c : TestisCell), c ∈ slice.cells ∧ 
    c.divisionType = CellDivisionType.Mitosis ∧
    c.phase = some MitosisPhase.Metaphase :=
  sorry

end metaphase_mitosis_observable_l1095_109543


namespace max_value_of_g_l1095_109565

/-- The quadratic function f(x, y) -/
def f (x y : ℝ) : ℝ := 10*x - 4*x^2 + 2*x*y

/-- The function g(x) is f(x, 3) -/
def g (x : ℝ) : ℝ := f x 3

theorem max_value_of_g :
  ∃ (m : ℝ), ∀ (x : ℝ), g x ≤ m ∧ ∃ (x₀ : ℝ), g x₀ = m ∧ m = 16 :=
sorry

end max_value_of_g_l1095_109565


namespace max_population_teeth_l1095_109547

theorem max_population_teeth (n : ℕ) (h : n = 32) :
  (2 : ℕ) ^ n = 4294967296 :=
sorry

end max_population_teeth_l1095_109547


namespace human_genome_project_satisfies_conditions_l1095_109557

/-- Represents a scientific plan --/
structure ScientificPlan where
  name : String
  launchYear : Nat
  participatingCountries : List String
  isMajorPlan : Bool

/-- The Human Genome Project --/
def humanGenomeProject : ScientificPlan := {
  name := "Human Genome Project",
  launchYear := 1990,
  participatingCountries := ["United States", "United Kingdom", "France", "Germany", "Japan", "China"],
  isMajorPlan := true
}

/-- The Manhattan Project --/
def manhattanProject : ScientificPlan := {
  name := "Manhattan Project",
  launchYear := 1942,
  participatingCountries := ["United States", "United Kingdom", "Canada"],
  isMajorPlan := true
}

/-- The Apollo Program --/
def apolloProgram : ScientificPlan := {
  name := "Apollo Program",
  launchYear := 1961,
  participatingCountries := ["United States"],
  isMajorPlan := true
}

/-- The set of "Three Major Scientific Plans" --/
def threeMajorPlans : List ScientificPlan := [humanGenomeProject, manhattanProject, apolloProgram]

/-- Theorem stating that the Human Genome Project satisfies all conditions --/
theorem human_genome_project_satisfies_conditions :
  humanGenomeProject.launchYear = 1990 ∧
  humanGenomeProject.participatingCountries = ["United States", "United Kingdom", "France", "Germany", "Japan", "China"] ∧
  humanGenomeProject ∈ threeMajorPlans := by
  sorry


end human_genome_project_satisfies_conditions_l1095_109557


namespace larger_number_proof_l1095_109588

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1000 → L = 10 * S + 10 → L = 1110 := by
sorry

end larger_number_proof_l1095_109588


namespace int_tan_triangle_unique_l1095_109518

/-- A triangle with integer tangents for all angles -/
structure IntTanTriangle where
  α : Real
  β : Real
  γ : Real
  sum_180 : α + β + γ = Real.pi
  tan_int_α : ∃ m : Int, Real.tan α = m
  tan_int_β : ∃ n : Int, Real.tan β = n
  tan_int_γ : ∃ k : Int, Real.tan γ = k

/-- The only possible combination of integer tangents for a triangle is (1, 2, 3) -/
theorem int_tan_triangle_unique (t : IntTanTriangle) :
  (Real.tan t.α = 1 ∧ Real.tan t.β = 2 ∧ Real.tan t.γ = 3) ∨
  (Real.tan t.α = 1 ∧ Real.tan t.β = 3 ∧ Real.tan t.γ = 2) ∨
  (Real.tan t.α = 2 ∧ Real.tan t.β = 1 ∧ Real.tan t.γ = 3) ∨
  (Real.tan t.α = 2 ∧ Real.tan t.β = 3 ∧ Real.tan t.γ = 1) ∨
  (Real.tan t.α = 3 ∧ Real.tan t.β = 1 ∧ Real.tan t.γ = 2) ∨
  (Real.tan t.α = 3 ∧ Real.tan t.β = 2 ∧ Real.tan t.γ = 1) :=
by sorry

end int_tan_triangle_unique_l1095_109518


namespace coordinates_of_A_min_length_AB_l1095_109524

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a line passing through the focus
structure LineThruFocus where
  slope : ℝ ⊕ PUnit  -- ℝ for finite slopes, PUnit for vertical line
  passes_thru_focus : True

-- Define the intersection points
def intersectionPoints (l : LineThruFocus) : PointOnParabola × PointOnParabola := sorry

-- Statement for part (1)
theorem coordinates_of_A (l : LineThruFocus) (A B : PointOnParabola) 
  (h : intersectionPoints l = (A, B)) (dist_AF : Real.sqrt ((A.x - 1)^2 + A.y^2) = 4) :
  (A.x = 3 ∧ A.y = 2 * Real.sqrt 3) ∨ (A.x = 3 ∧ A.y = -2 * Real.sqrt 3) := sorry

-- Statement for part (2)
theorem min_length_AB : 
  ∃ (min_length : ℝ), ∀ (l : LineThruFocus) (A B : PointOnParabola),
    intersectionPoints l = (A, B) → 
    Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) ≥ min_length ∧
    min_length = 4 := sorry

end coordinates_of_A_min_length_AB_l1095_109524


namespace perpendicular_slope_l1095_109515

/-- The slope of a line perpendicular to the line passing through (3, -4) and (-2, 5) is 5/9 -/
theorem perpendicular_slope : 
  let x₁ : ℚ := 3
  let y₁ : ℚ := -4
  let x₂ : ℚ := -2
  let y₂ : ℚ := 5
  let m : ℚ := (y₂ - y₁) / (x₂ - x₁)
  (- (1 / m)) = 5 / 9 := by sorry

end perpendicular_slope_l1095_109515
