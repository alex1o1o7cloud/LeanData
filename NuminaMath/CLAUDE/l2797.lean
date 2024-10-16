import Mathlib

namespace NUMINAMATH_CALUDE_functional_equation_solution_l2797_279743

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → f (x * f y) = f (x * y) + x) →
  (∀ x : ℝ, x > 0 → f x = x + 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2797_279743


namespace NUMINAMATH_CALUDE_vs_length_l2797_279707

/-- A square piece of paper PQRS with side length 8 cm is folded so that corner R 
    coincides with T, the midpoint of PS. The crease UV intersects RS at V. -/
structure FoldedSquare where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point R -/
  R : ℝ × ℝ
  /-- Point S -/
  S : ℝ × ℝ
  /-- Point T (midpoint of PS) -/
  T : ℝ × ℝ
  /-- Point V (intersection of UV and RS) -/
  V : ℝ × ℝ
  /-- PQRS forms a square with side length 8 -/
  square_constraint : 
    P.1 = Q.1 ∧ Q.2 = R.2 ∧ R.1 = S.1 ∧ S.2 = P.2 ∧
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side_length^2 ∧
    side_length = 8
  /-- T is the midpoint of PS -/
  midpoint_constraint : T = ((P.1 + S.1) / 2, (P.2 + S.2) / 2)
  /-- V is on RS -/
  v_on_rs : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ V = (R.1 * (1 - t) + S.1 * t, R.2 * (1 - t) + S.2 * t)
  /-- Distance RV equals distance TV (fold constraint) -/
  fold_constraint : (R.1 - V.1)^2 + (R.2 - V.2)^2 = (T.1 - V.1)^2 + (T.2 - V.2)^2

/-- The length of VS in the folded square is 3 cm -/
theorem vs_length (fs : FoldedSquare) : 
  ((fs.V.1 - fs.S.1)^2 + (fs.V.2 - fs.S.2)^2)^(1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_vs_length_l2797_279707


namespace NUMINAMATH_CALUDE_conditional_probability_first_class_l2797_279774

/-- A box containing products -/
structure Box where
  total : Nat
  firstClass : Nat
  secondClass : Nat
  h_sum : firstClass + secondClass = total

/-- The probability of selecting a first-class product on the second draw
    given that a first-class product was selected on the first draw -/
def conditionalProbability (b : Box) : ℚ :=
  (b.firstClass - 1 : ℚ) / (b.total - 1 : ℚ)

theorem conditional_probability_first_class
  (b : Box)
  (h_total : b.total = 4)
  (h_first : b.firstClass = 3)
  (h_second : b.secondClass = 1) :
  conditionalProbability b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_first_class_l2797_279774


namespace NUMINAMATH_CALUDE_close_interval_is_two_to_three_l2797_279725

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the close function property
def is_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Theorem statement
theorem close_interval_is_two_to_three :
  ∀ a b : ℝ, a ≤ 2 ∧ 3 ≤ b → (is_close f g a b ↔ a = 2 ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_close_interval_is_two_to_three_l2797_279725


namespace NUMINAMATH_CALUDE_firecracker_explosion_speed_l2797_279777

/-- The speed of a fragment after an explosion, given initial conditions of a firecracker. -/
theorem firecracker_explosion_speed 
  (v₀ : ℝ)           -- Initial upward speed of firecracker
  (t : ℝ)            -- Time of explosion
  (m₁ m₂ : ℝ)        -- Masses of fragments
  (v_small : ℝ)      -- Horizontal speed of smaller fragment after explosion
  (g : ℝ)            -- Acceleration due to gravity
  (h : v₀ = 20)      -- Initial speed is 20 m/s
  (h_t : t = 3)      -- Explosion occurs at 3 seconds
  (h_m : m₂ = 2 * m₁) -- Mass ratio is 1:2
  (h_v : v_small = 16) -- Smaller fragment's horizontal speed is 16 m/s
  (h_g : g = 10)     -- Acceleration due to gravity is 10 m/s^2
  : ∃ v : ℝ, v = 17 ∧ v = 
    Real.sqrt ((2 * m₁ * v_small / (m₁ + m₂))^2 + (v₀ - g * t)^2) :=
by sorry

end NUMINAMATH_CALUDE_firecracker_explosion_speed_l2797_279777


namespace NUMINAMATH_CALUDE_opposite_black_is_orange_l2797_279761

-- Define the colors
inductive Color
| Orange | Yellow | Blue | Pink | Violet | Black

-- Define a cube face
structure Face :=
  (color : Color)

-- Define a cube
structure Cube :=
  (top : Face)
  (front : Face)
  (right : Face)
  (bottom : Face)
  (back : Face)
  (left : Face)

-- Define the views
def view1 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Blue ∧ c.right.color = Color.Pink

def view2 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Violet ∧ c.right.color = Color.Pink

def view3 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Yellow ∧ c.right.color = Color.Pink

-- Theorem statement
theorem opposite_black_is_orange (c : Cube) :
  view1 c → view2 c → view3 c → c.bottom.color = Color.Black →
  c.top.color = Color.Orange :=
sorry

end NUMINAMATH_CALUDE_opposite_black_is_orange_l2797_279761


namespace NUMINAMATH_CALUDE_dog_legs_on_street_l2797_279799

theorem dog_legs_on_street (total_animals : ℕ) (cat_fraction : ℚ) (dog_legs : ℕ) : 
  total_animals = 300 →
  cat_fraction = 2 / 3 →
  dog_legs = 4 →
  (total_animals * (1 - cat_fraction) : ℚ).num * dog_legs = 400 :=
by sorry

end NUMINAMATH_CALUDE_dog_legs_on_street_l2797_279799


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2797_279727

theorem square_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = Real.sqrt 2020) :
  x^2 + 1/x^2 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2797_279727


namespace NUMINAMATH_CALUDE_abs_z_equals_10_l2797_279711

def z : ℂ := (3 + Complex.I)^2 * Complex.I

theorem abs_z_equals_10 : Complex.abs z = 10 := by sorry

end NUMINAMATH_CALUDE_abs_z_equals_10_l2797_279711


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2797_279748

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) :
  Nat.lcm X Y = 180 →
  (X : ℚ) / (Y : ℚ) = 2 / 5 →
  Nat.gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2797_279748


namespace NUMINAMATH_CALUDE_gas_station_sales_l2797_279792

/-- The total number of boxes sold at a gas station -/
def total_boxes (chocolate_boxes sugar_boxes gum_boxes : ℕ) : ℕ :=
  chocolate_boxes + sugar_boxes + gum_boxes

/-- Theorem: The gas station sold 9 boxes in total -/
theorem gas_station_sales : total_boxes 2 5 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gas_station_sales_l2797_279792


namespace NUMINAMATH_CALUDE_thirty_percent_more_than_75_l2797_279731

theorem thirty_percent_more_than_75 (x : ℝ) : x / 2 = 75 * 1.3 → x = 195 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_more_than_75_l2797_279731


namespace NUMINAMATH_CALUDE_lattice_points_bound_l2797_279742

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  area : ℝ
  semiperimeter : ℝ
  lattice_points : ℕ

/-- Theorem: For any convex figure, the number of lattice points inside
    is greater than the difference between its area and semiperimeter -/
theorem lattice_points_bound (figure : ConvexFigure) :
  figure.lattice_points > figure.area - figure.semiperimeter := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_bound_l2797_279742


namespace NUMINAMATH_CALUDE_average_temperature_l2797_279703

def temperatures : List ℤ := [-36, 13, -15, -10]

theorem average_temperature (temps := temperatures) :
  (temps.sum : ℚ) / temps.length = -12 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l2797_279703


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2797_279788

/-- A quadratic function f(x) = x^2 + mx + n with roots -2 and -1 -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_function_properties (m n : ℝ) :
  (∀ x, f m n x = 0 ↔ x = -2 ∨ x = -1) →
  (m = 3 ∧ n = 2) ∧
  (∀ x ∈ Set.Icc (-5 : ℝ) 5,
    f m n x ≥ -1/4 ∧
    f m n x ≤ 42 ∧
    (∃ x₁ ∈ Set.Icc (-5 : ℝ) 5, f m n x₁ = -1/4) ∧
    (∃ x₂ ∈ Set.Icc (-5 : ℝ) 5, f m n x₂ = 42)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2797_279788


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l2797_279796

/-- The function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition f(f(x)) -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = (11 - √13)/2 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f_comp c r₁ = 0 ∧ f_comp c r₂ = 0 ∧ f_comp c r₃ = 0) ↔
  c = (11 - Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l2797_279796


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2797_279755

theorem sqrt_equation_solution : 
  {x : ℝ | Real.sqrt (2*x - 4) - Real.sqrt (x + 5) = 1} = {4, 20} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2797_279755


namespace NUMINAMATH_CALUDE_spinster_cat_ratio_l2797_279763

theorem spinster_cat_ratio :
  ∀ (s c : ℕ),
    s = 12 →
    c = s + 42 →
    ∃ (n : ℕ), n * s = 2 * c ∧ 9 * s = n * c :=
by
  sorry

end NUMINAMATH_CALUDE_spinster_cat_ratio_l2797_279763


namespace NUMINAMATH_CALUDE_problem_solution_l2797_279706

def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x + x > 0 ↔ (-3 < x ∧ x < 1) ∨ x > 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3) ↔ -5 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2797_279706


namespace NUMINAMATH_CALUDE_standard_of_living_purchasing_power_correlated_l2797_279771

/-- Represents a person's standard of living -/
def StandardOfLiving : Type := ℝ

/-- Represents a person's purchasing power -/
def PurchasingPower : Type := ℝ

/-- Definition of correlation as a statistical relationship between two random variables -/
def Correlated (X Y : Type) : Prop := sorry

/-- Theorem stating that standard of living and purchasing power are correlated -/
theorem standard_of_living_purchasing_power_correlated :
  Correlated StandardOfLiving PurchasingPower :=
sorry

end NUMINAMATH_CALUDE_standard_of_living_purchasing_power_correlated_l2797_279771


namespace NUMINAMATH_CALUDE_article_selling_price_l2797_279765

def cost_price : ℝ := 250
def profit_percentage : ℝ := 0.60

def selling_price : ℝ := cost_price + (profit_percentage * cost_price)

theorem article_selling_price : selling_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_article_selling_price_l2797_279765


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2797_279772

-- Define the universal set U
def U : Finset Nat := {2, 3, 4, 5, 6}

-- Define set A
def A : Finset Nat := {2, 5, 6}

-- Define set B
def B : Finset Nat := {3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ B) ∩ A = {2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2797_279772


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2797_279714

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4 / (1 - r))) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2797_279714


namespace NUMINAMATH_CALUDE_stadium_seats_count_l2797_279769

/-- The number of seats in a stadium is equal to the sum of occupied and empty seats -/
theorem stadium_seats_count
  (children : ℕ)
  (adults : ℕ)
  (empty_seats : ℕ)
  (h1 : children = 52)
  (h2 : adults = 29)
  (h3 : empty_seats = 14) :
  children + adults + empty_seats = 95 := by
  sorry

#check stadium_seats_count

end NUMINAMATH_CALUDE_stadium_seats_count_l2797_279769


namespace NUMINAMATH_CALUDE_tan_value_from_double_angle_formula_l2797_279759

theorem tan_value_from_double_angle_formula (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin (2 * θ) = 2 - 2 * Real.cos (2 * θ)) : 
  Real.tan θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_double_angle_formula_l2797_279759


namespace NUMINAMATH_CALUDE_acid_dilution_l2797_279747

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 40 ∧ 
  initial_concentration = 0.4 ∧ 
  water_added = 24 ∧ 
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_dilution_l2797_279747


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2797_279766

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2797_279766


namespace NUMINAMATH_CALUDE_ab_equals_seventeen_l2797_279750

theorem ab_equals_seventeen
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2*a - b)
  : a * b = 17 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_seventeen_l2797_279750


namespace NUMINAMATH_CALUDE_circular_seating_nine_seven_l2797_279740

/-- The number of ways to choose 7 people from 9 and seat them around a circular table -/
def circular_seating_arrangements (total_people : ℕ) (seats : ℕ) : ℕ :=
  (total_people.choose (total_people - seats)) * (seats - 1).factorial

theorem circular_seating_nine_seven :
  circular_seating_arrangements 9 7 = 25920 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_nine_seven_l2797_279740


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_2016_l2797_279738

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the expression
def expression : ℚ :=
  (factorial 2017 + factorial 2014) / (factorial 2016 + factorial 2015)

-- Theorem statement
theorem floor_of_expression_equals_2016 :
  ⌊expression⌋ = 2016 := by sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_2016_l2797_279738


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l2797_279791

/-- Given the scoring information for two basketball teams, prove the ratio of 2-pointers scored by the opponents to Mark's team. -/
theorem basketball_score_ratio :
  let marks_two_pointers : ℕ := 25
  let marks_three_pointers : ℕ := 8
  let marks_free_throws : ℕ := 10
  let opponents_three_pointers : ℕ := marks_three_pointers / 2
  let opponents_free_throws : ℕ := marks_free_throws / 2
  let total_points : ℕ := 201
  ∃ (x : ℚ),
    (2 * marks_two_pointers + 3 * marks_three_pointers + marks_free_throws) +
    (2 * (x * marks_two_pointers) + 3 * opponents_three_pointers + opponents_free_throws) = total_points ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_ratio_l2797_279791


namespace NUMINAMATH_CALUDE_slant_height_and_height_not_unique_l2797_279726

/-- Represents a right triangular pyramid with a square base -/
structure RightTriangularPyramid where
  base_side : ℝ
  height : ℝ
  slant_height : ℝ

/-- Predicate to check if two pyramids are different -/
def different_pyramids (p1 p2 : RightTriangularPyramid) : Prop :=
  p1.base_side ≠ p2.base_side ∧ p1.height = p2.height ∧ p1.slant_height = p2.slant_height

/-- Theorem stating that slant height and height do not uniquely specify the pyramid -/
theorem slant_height_and_height_not_unique :
  ∃ (p1 p2 : RightTriangularPyramid), different_pyramids p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_slant_height_and_height_not_unique_l2797_279726


namespace NUMINAMATH_CALUDE_wedding_attendance_l2797_279705

/-- The number of people Laura invited to her wedding. -/
def invited : ℕ := 220

/-- The percentage of people who typically don't show up. -/
def no_show_percentage : ℚ := 5 / 100

/-- The number of people expected to attend Laura's wedding. -/
def expected_attendance : ℕ := 209

/-- Proves that the expected attendance at Laura's wedding is 209 people. -/
theorem wedding_attendance : 
  (invited : ℚ) * (1 - no_show_percentage) = expected_attendance := by
  sorry

end NUMINAMATH_CALUDE_wedding_attendance_l2797_279705


namespace NUMINAMATH_CALUDE_show_attendance_l2797_279778

theorem show_attendance (adult_price child_price total_cost : ℕ) 
  (num_children : ℕ) (h1 : adult_price = 12) (h2 : child_price = 10) 
  (h3 : num_children = 3) (h4 : total_cost = 66) : 
  ∃ (num_adults : ℕ), num_adults = 3 ∧ 
    adult_price * num_adults + child_price * num_children = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_show_attendance_l2797_279778


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_ten_l2797_279717

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of a tetrahedron PQRS with given edge lengths is 10 -/
theorem tetrahedron_volume_is_ten :
  let t : Tetrahedron := {
    pq := 3,
    pr := 5,
    ps := 6,
    qr := 4,
    qs := Real.sqrt 26,
    rs := 5
  }
  tetrahedronVolume t = 10 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_is_ten_l2797_279717


namespace NUMINAMATH_CALUDE_final_student_count_l2797_279756

/-- The number of students in Beth's class at different stages --/
def students_in_class (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the final number of students in Beth's class --/
theorem final_student_count :
  students_in_class 150 30 15 = 165 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l2797_279756


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2797_279722

/-- Simplification of a polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  3 * x + 10 * x^2 + 5 * x^3 + 15 - (7 - 3 * x - 10 * x^2 - 5 * x^3) =
  10 * x^3 + 20 * x^2 + 6 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2797_279722


namespace NUMINAMATH_CALUDE_seed_germination_requires_water_l2797_279704

-- Define a seed
structure Seed where
  water_content : ℝ
  germinated : Bool

-- Define the germination process
def germinate (s : Seed) : Prop :=
  s.germinated = true

-- Theorem: A seed cannot germinate without water
theorem seed_germination_requires_water (s : Seed) :
  germinate s → s.water_content > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_seed_germination_requires_water_l2797_279704


namespace NUMINAMATH_CALUDE_race_time_difference_l2797_279734

def malcolm_speed : ℝ := 5.5
def joshua_speed : ℝ := 7.5
def race_distance : ℝ := 12

theorem race_time_difference : 
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l2797_279734


namespace NUMINAMATH_CALUDE_simplify_fraction_l2797_279729

theorem simplify_fraction : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2797_279729


namespace NUMINAMATH_CALUDE_train_length_l2797_279744

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 126 → time_s = 16 → speed_kmh * (5/18) * time_s = 560 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2797_279744


namespace NUMINAMATH_CALUDE_parallelogram_area_l2797_279741

/-- The area of a parallelogram with base 15 and height 5 is 75 square feet. -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 15 ∧ height = 5 → area = base * height → area = 75

/-- Proof of the parallelogram area theorem -/
lemma prove_parallelogram_area : parallelogram_area 15 5 75 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2797_279741


namespace NUMINAMATH_CALUDE_integral_polynomial_l2797_279723

variables (a b c p : ℝ) (x : ℝ)

theorem integral_polynomial (a b c p : ℝ) (x : ℝ) :
  deriv (fun x => (a/4) * x^4 + (b/3) * x^3 + (c/2) * x^2 + p * x) x
  = a * x^3 + b * x^2 + c * x + p :=
by sorry

end NUMINAMATH_CALUDE_integral_polynomial_l2797_279723


namespace NUMINAMATH_CALUDE_prime_quadratic_equation_solution_l2797_279712

theorem prime_quadratic_equation_solution (a b Q R : ℕ) : 
  Nat.Prime a → 
  Nat.Prime b → 
  a ≠ b → 
  a^2 - a*Q + R = 0 → 
  b^2 - b*Q + R = 0 → 
  R = 6 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_equation_solution_l2797_279712


namespace NUMINAMATH_CALUDE_sum_reciprocals_equal_negative_two_l2797_279724

theorem sum_reciprocals_equal_negative_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y + x * y = 0) : y / x + x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equal_negative_two_l2797_279724


namespace NUMINAMATH_CALUDE_marys_friends_ages_sum_l2797_279746

theorem marys_friends_ages_sum : 
  ∀ (a b c d : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit positive integers
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct
    ((a * b = 28 ∧ c * d = 45) ∨ (a * c = 28 ∧ b * d = 45) ∨ 
     (a * d = 28 ∧ b * c = 45) ∨ (b * c = 28 ∧ a * d = 45) ∨ 
     (b * d = 28 ∧ a * c = 45) ∨ (c * d = 28 ∧ a * b = 45)) →
    a + b + c + d = 25 := by
  sorry

end NUMINAMATH_CALUDE_marys_friends_ages_sum_l2797_279746


namespace NUMINAMATH_CALUDE_arun_age_is_60_l2797_279732

/-- Proves that Arun's age is 60 years given the conditions from the problem -/
theorem arun_age_is_60 (arun_age madan_age gokul_age : ℕ) 
  (h1 : (arun_age - 6) / 18 = gokul_age)
  (h2 : gokul_age = madan_age - 2)
  (h3 : madan_age = 5) : 
  arun_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_arun_age_is_60_l2797_279732


namespace NUMINAMATH_CALUDE_exactly_one_true_proposition_l2797_279737

open Real

theorem exactly_one_true_proposition :
  let prop1 := ∀ x : ℝ, x^4 > x^2
  let prop2 := ∀ p q : Prop, (¬(p ∧ q)) → (¬p ∧ ¬q)
  let prop3 := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)
  (¬prop1 ∧ ¬prop2 ∧ prop3) :=
by sorry

#check exactly_one_true_proposition

end NUMINAMATH_CALUDE_exactly_one_true_proposition_l2797_279737


namespace NUMINAMATH_CALUDE_isosceles_triangle_legs_l2797_279709

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2

-- Define the theorem
theorem isosceles_triangle_legs (t : IsoscelesTriangle) :
  t.side1 + t.side2 + t.base = 18 ∧ (t.side1 = 8 ∨ t.base = 8) →
  t.side1 = 8 ∨ t.side1 = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_legs_l2797_279709


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l2797_279713

def Digits : Finset Nat := {1, 2, 3, 4}

theorem three_digit_numbers_count : 
  Finset.card (Finset.product (Finset.product Digits Digits) Digits) = 64 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l2797_279713


namespace NUMINAMATH_CALUDE_min_workers_theorem_l2797_279733

/-- Represents the company's keychain production and sales model -/
structure KeychainCompany where
  maintenance_fee : ℝ  -- Daily maintenance fee
  worker_wage : ℝ      -- Hourly wage per worker
  keychains_per_hour : ℝ  -- Keychains produced per worker per hour
  keychain_price : ℝ   -- Price of each keychain
  work_hours : ℝ       -- Hours in a workday

/-- Calculates the minimum number of workers needed for profit -/
def min_workers_for_profit (company : KeychainCompany) : ℕ :=
  sorry

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_theorem (company : KeychainCompany) 
  (h1 : company.maintenance_fee = 500)
  (h2 : company.worker_wage = 15)
  (h3 : company.keychains_per_hour = 5)
  (h4 : company.keychain_price = 3.10)
  (h5 : company.work_hours = 8) :
  min_workers_for_profit company = 126 :=
sorry

end NUMINAMATH_CALUDE_min_workers_theorem_l2797_279733


namespace NUMINAMATH_CALUDE_power_difference_equality_l2797_279784

theorem power_difference_equality : (3^2)^3 - (2^3)^2 = 665 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l2797_279784


namespace NUMINAMATH_CALUDE_scientific_notation_of_132000000_l2797_279721

theorem scientific_notation_of_132000000 :
  (132000000 : ℝ) = 1.32 * (10 : ℝ) ^ 8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_132000000_l2797_279721


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2797_279735

/-- Represents the shape created by arranging 8 unit cubes -/
structure CubeShape where
  center_cube : Unit
  surrounding_cubes : Fin 6 → Unit
  top_cube : Unit

/-- Calculates the volume of the CubeShape -/
def volume (shape : CubeShape) : ℕ := 8

/-- Calculates the surface area of the CubeShape -/
def surface_area (shape : CubeShape) : ℕ := 28

/-- Theorem stating that the ratio of volume to surface area is 2/7 -/
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  (volume shape : ℚ) / (surface_area shape : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2797_279735


namespace NUMINAMATH_CALUDE_problem_solution_l2797_279702

theorem problem_solution (x y : ℝ) (h1 : 15 * x = x + 280) (h2 : y = x^2 + 5*x - 12) :
  x = 20 ∧ y = 488 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2797_279702


namespace NUMINAMATH_CALUDE_tangent_slope_at_A_l2797_279776

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + x

-- Define the point A
def point_A : ℝ × ℝ := (2, 6)

-- Theorem statement
theorem tangent_slope_at_A :
  (deriv f) point_A.1 = 5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_A_l2797_279776


namespace NUMINAMATH_CALUDE_banana_apple_equivalence_l2797_279787

-- Define the worth of bananas in terms of apples
def banana_worth (b : ℚ) : ℚ := 
  (12 : ℚ) / ((3 / 4) * 16)

-- Theorem statement
theorem banana_apple_equivalence : 
  banana_worth ((1 / 3) * 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_apple_equivalence_l2797_279787


namespace NUMINAMATH_CALUDE_thirteen_points_guarantee_win_thirteen_smallest_guarantee_l2797_279700

/-- Represents the points awarded for each position in a race -/
def race_points : Fin 3 → ℕ
  | 0 => 5  -- First place
  | 1 => 3  -- Second place
  | 2 => 1  -- Third place
  | _ => 0  -- This case should never occur due to Fin 3

/-- The total number of races -/
def num_races : ℕ := 3

/-- A function to calculate the maximum points possible for the second-place student -/
def max_second_place_points : ℕ := sorry

/-- Theorem stating that 13 points guarantees more points than any other student -/
theorem thirteen_points_guarantee_win :
  ∀ (student_points : ℕ),
    student_points ≥ 13 →
    student_points > max_second_place_points :=
  sorry

/-- Theorem stating that 13 is the smallest number of points that guarantees a win -/
theorem thirteen_smallest_guarantee :
  ∀ (n : ℕ),
    n < 13 →
    ∃ (other_points : ℕ),
      other_points ≥ n ∧
      other_points ≤ max_second_place_points :=
  sorry

end NUMINAMATH_CALUDE_thirteen_points_guarantee_win_thirteen_smallest_guarantee_l2797_279700


namespace NUMINAMATH_CALUDE_davids_math_marks_l2797_279762

/-- Calculates the marks in an unknown subject given the marks in other subjects and the average --/
def calculate_unknown_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + physics + chemistry + biology)

theorem davids_math_marks :
  let english := 81
  let physics := 82
  let chemistry := 67
  let biology := 85
  let average := 76
  calculate_unknown_marks english physics chemistry biology average = 65 := by
  sorry

#eval calculate_unknown_marks 81 82 67 85 76

end NUMINAMATH_CALUDE_davids_math_marks_l2797_279762


namespace NUMINAMATH_CALUDE_white_area_theorem_l2797_279789

/-- A painting with two white squares in a gray field -/
structure Painting where
  s : ℝ
  gray_area : ℝ
  total_side_length : ℝ
  smaller_square_side : ℝ
  larger_square_side : ℝ

/-- The theorem stating the area of the white part given the conditions -/
theorem white_area_theorem (p : Painting) 
    (h1 : p.total_side_length = 6 * p.s)
    (h2 : p.smaller_square_side = p.s)
    (h3 : p.larger_square_side = 2 * p.s)
    (h4 : p.gray_area = 62) : 
  ∃ (white_area : ℝ), white_area = 10 := by
  sorry

end NUMINAMATH_CALUDE_white_area_theorem_l2797_279789


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l2797_279718

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l2797_279718


namespace NUMINAMATH_CALUDE_sixth_term_value_l2797_279775

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sixth_term_value (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : arithmetic_sequence a)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = 2) :
  a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2797_279775


namespace NUMINAMATH_CALUDE_second_fragment_speed_is_52_l2797_279783

/-- Represents the motion of a firecracker that explodes into two fragments -/
structure Firecracker where
  initial_speed : ℝ
  explosion_time : ℝ
  gravity : ℝ
  first_fragment_horizontal_speed : ℝ

/-- Calculates the speed of the second fragment after explosion -/
def second_fragment_speed (f : Firecracker) : ℝ :=
  sorry

/-- Theorem stating that the speed of the second fragment is 52 m/s -/
theorem second_fragment_speed_is_52 (f : Firecracker) 
  (h1 : f.initial_speed = 20)
  (h2 : f.explosion_time = 3)
  (h3 : f.gravity = 10)
  (h4 : f.first_fragment_horizontal_speed = 48) :
  second_fragment_speed f = 52 :=
  sorry

end NUMINAMATH_CALUDE_second_fragment_speed_is_52_l2797_279783


namespace NUMINAMATH_CALUDE_units_digit_of_product_with_sum_factorials_l2797_279701

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product_with_sum_factorials : 
  units_digit (7 * sum_factorials 2023) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_with_sum_factorials_l2797_279701


namespace NUMINAMATH_CALUDE_option_C_most_suitable_for_comprehensive_survey_l2797_279760

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Understanding the sleep time of middle school students nationwide
  | B  -- Understanding the water quality of a river
  | C  -- Surveying the vision of all classmates
  | D  -- Understanding the service life of a batch of light bulbs

/-- Defines what makes a survey comprehensive -/
def isComprehensive (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that option C is the most suitable for a comprehensive survey -/
theorem option_C_most_suitable_for_comprehensive_survey :
  ∀ (option : SurveyOption), isComprehensive option → option = SurveyOption.C :=
by sorry

end NUMINAMATH_CALUDE_option_C_most_suitable_for_comprehensive_survey_l2797_279760


namespace NUMINAMATH_CALUDE_plane_distance_proof_l2797_279745

-- Define the plane's speed in still air
def plane_speed : ℝ := 262.5

-- Define the time taken with tail wind
def time_with_wind : ℝ := 3

-- Define the time taken against wind
def time_against_wind : ℝ := 4

-- Define the wind speed (to be solved)
def wind_speed : ℝ := 37.5

-- Define the distance (to be proved)
def distance : ℝ := 900

-- Theorem statement
theorem plane_distance_proof :
  distance = (plane_speed + wind_speed) * time_with_wind ∧
  distance = (plane_speed - wind_speed) * time_against_wind :=
by sorry

end NUMINAMATH_CALUDE_plane_distance_proof_l2797_279745


namespace NUMINAMATH_CALUDE_power_sum_seven_l2797_279720

theorem power_sum_seven (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 6)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 14) :
  α₁^7 + α₂^7 + α₃^7 = 46 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_seven_l2797_279720


namespace NUMINAMATH_CALUDE_library_books_count_l2797_279749

theorem library_books_count : ∀ (total_books : ℕ), 
  (35 : ℚ) / 100 * total_books + 104 = total_books → total_books = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l2797_279749


namespace NUMINAMATH_CALUDE_one_lamp_position_l2797_279758

/-- Represents a position on the 5x5 grid -/
structure Position where
  x : Fin 5
  y : Fin 5

/-- Represents the state of the 5x5 grid of lamps -/
def Grid := Fin 5 → Fin 5 → Bool

/-- The operation of toggling a lamp and its adjacent lamps -/
def toggle (grid : Grid) (pos : Position) : Grid := sorry

/-- Checks if only one lamp is on in the grid -/
def onlyOneLampOn (grid : Grid) : Bool := sorry

/-- Checks if a position is either the center or directly diagonal to the center -/
def isCenterOrDiagonal (pos : Position) : Bool := sorry

/-- The main theorem: If only one lamp is on after a sequence of toggle operations,
    it must be in the center or directly diagonal to the center -/
theorem one_lamp_position (grid : Grid) (pos : Position) :
  (∃ (ops : List Position), onlyOneLampOn (ops.foldl toggle grid)) →
  (onlyOneLampOn grid ∧ grid pos.x pos.y = true) →
  isCenterOrDiagonal pos := sorry

end NUMINAMATH_CALUDE_one_lamp_position_l2797_279758


namespace NUMINAMATH_CALUDE_pell_equation_solutions_l2797_279786

theorem pell_equation_solutions :
  let solutions : List (ℤ × ℤ) := [(2, 1), (7, 4), (26, 15), (97, 56)]
  ∀ (x y : ℤ), (x, y) ∈ solutions → x^2 - 3*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_pell_equation_solutions_l2797_279786


namespace NUMINAMATH_CALUDE_exponent_simplification_l2797_279728

theorem exponent_simplification : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2797_279728


namespace NUMINAMATH_CALUDE_mersenne_primes_less_than_1000_are_3_7_31_127_l2797_279770

def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1

def mersenne_primes_less_than_1000 : Set ℕ :=
  {p : ℕ | is_mersenne_prime p ∧ p < 1000}

theorem mersenne_primes_less_than_1000_are_3_7_31_127 :
  mersenne_primes_less_than_1000 = {3, 7, 31, 127} :=
by sorry

end NUMINAMATH_CALUDE_mersenne_primes_less_than_1000_are_3_7_31_127_l2797_279770


namespace NUMINAMATH_CALUDE_translation_sum_l2797_279708

/-- Given two points P and Q in a 2D plane, where P is translated m units left
    and n units up to obtain Q, prove that m + n = 4. -/
theorem translation_sum (P Q : ℝ × ℝ) (m n : ℝ) : 
  P = (-1, -3) → Q = (-2, 0) → Q.1 = P.1 - m → Q.2 = P.2 + n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_translation_sum_l2797_279708


namespace NUMINAMATH_CALUDE_total_cost_is_48_l2797_279739

/-- The cost of a pencil case in yuan -/
def pencil_case_cost : ℕ := 8

/-- The cost of a backpack in yuan -/
def backpack_cost : ℕ := 5 * pencil_case_cost

/-- The total cost of a backpack and a pencil case in yuan -/
def total_cost : ℕ := backpack_cost + pencil_case_cost

/-- Theorem stating that the total cost of a backpack and a pencil case is 48 yuan -/
theorem total_cost_is_48 : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_48_l2797_279739


namespace NUMINAMATH_CALUDE_unique_factorial_sum_number_l2797_279794

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = Nat.factorial a + Nat.factorial b + Nat.factorial c

theorem unique_factorial_sum_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 145 :=
sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_number_l2797_279794


namespace NUMINAMATH_CALUDE_remaining_water_is_one_cup_l2797_279736

/-- Represents Harry's hike and water consumption --/
structure HikeData where
  total_distance : ℝ
  initial_water : ℝ
  duration : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_miles_rate : ℝ

/-- Calculates the remaining water after the hike --/
def remaining_water (data : HikeData) : ℝ :=
  data.initial_water
  - (data.first_miles_rate * (data.total_distance - 1))
  - data.last_mile_consumption
  - (data.leak_rate * data.duration)

/-- Theorem stating that the remaining water is 1 cup --/
theorem remaining_water_is_one_cup (data : HikeData)
  (h1 : data.total_distance = 7)
  (h2 : data.initial_water = 9)
  (h3 : data.duration = 2)
  (h4 : data.leak_rate = 1)
  (h5 : data.last_mile_consumption = 2)
  (h6 : data.first_miles_rate = 0.6666666666666666)
  : remaining_water data = 1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_is_one_cup_l2797_279736


namespace NUMINAMATH_CALUDE_probability_even_product_l2797_279797

def setA : Finset ℕ := {1, 2, 3, 4}
def setB : Finset ℕ := {5, 6, 7, 8}

def isEven (n : ℕ) : Bool := n % 2 = 0

def evenProductPairs : Finset (ℕ × ℕ) :=
  setA.product setB |>.filter (fun (a, b) => isEven (a * b))

theorem probability_even_product :
  (evenProductPairs.card : ℚ) / ((setA.card * setB.card) : ℚ) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_product_l2797_279797


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l2797_279716

theorem relationship_between_x_and_y 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (x : ℝ) 
  (hx : x = Real.sqrt (a + b) - Real.sqrt b) 
  (y : ℝ) 
  (hy : y = Real.sqrt b - Real.sqrt (b - a)) : 
  x < y := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l2797_279716


namespace NUMINAMATH_CALUDE_refrigerator_theorem_l2797_279790

def refrigerator_problem (P : ℝ) : Prop :=
  let discount_rate : ℝ := 0.20
  let profit_rate : ℝ := 0.10
  let additional_costs : ℝ := 375
  let selling_price : ℝ := 18975
  let purchase_price : ℝ := P * (1 - discount_rate)
  let total_price : ℝ := purchase_price + additional_costs
  (P * (1 + profit_rate) = selling_price) → (total_price = 14175)

theorem refrigerator_theorem :
  ∃ P : ℝ, refrigerator_problem P :=
sorry

end NUMINAMATH_CALUDE_refrigerator_theorem_l2797_279790


namespace NUMINAMATH_CALUDE_quadratic_root_conditions_l2797_279793

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 + (a^2 + 1) * x + a - 2

-- Define the roots of the quadratic equation
def roots (a : ℝ) : Set ℝ := {x : ℝ | quadratic a x = 0}

-- Theorem statement
theorem quadratic_root_conditions (a : ℝ) :
  (∃ x ∈ roots a, x > 1) ∧ (∃ y ∈ roots a, y < -1) → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_conditions_l2797_279793


namespace NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l2797_279751

-- Define a spatial geometric body
structure SpatialBody where
  name : String

-- Define the front view of a spatial body
inductive FrontView
  | Triangle
  | Rectangle
  | Other

-- Define a function that returns the front view of a spatial body
def frontViewOf (body : SpatialBody) : FrontView :=
  sorry

-- Define a cylinder
def cylinder : SpatialBody :=
  { name := "Cylinder" }

-- Theorem: A cylinder cannot have a triangular front view
theorem cylinder_not_triangular_front_view :
  frontViewOf cylinder ≠ FrontView.Triangle :=
sorry

end NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l2797_279751


namespace NUMINAMATH_CALUDE_watch_cost_l2797_279798

theorem watch_cost (watch_cost strap_cost : ℝ) 
  (total_cost : watch_cost + strap_cost = 120)
  (cost_difference : watch_cost = strap_cost + 100) :
  watch_cost = 110 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_l2797_279798


namespace NUMINAMATH_CALUDE_apples_distribution_l2797_279768

/-- The number of people who received apples -/
def num_people (total_apples : ℕ) (apples_per_person : ℚ) : ℚ :=
  total_apples / apples_per_person

/-- Proof that 3 people received apples -/
theorem apples_distribution (total_apples : ℕ) (apples_per_person : ℚ) 
  (h1 : total_apples = 45)
  (h2 : apples_per_person = 15.0) : 
  num_people total_apples apples_per_person = 3 := by
  sorry


end NUMINAMATH_CALUDE_apples_distribution_l2797_279768


namespace NUMINAMATH_CALUDE_two_round_trips_time_l2797_279779

/-- Represents the time for a round trip given the time for one-way trip at normal speed -/
def round_trip_time (one_way_time : ℝ) : ℝ := one_way_time + 2 * one_way_time

/-- Proves that two round trips take 6 hours when one-way trip takes 1 hour -/
theorem two_round_trips_time : round_trip_time 1 * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_round_trips_time_l2797_279779


namespace NUMINAMATH_CALUDE_equal_chords_subtend_equal_arcs_l2797_279752

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord in a circle -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- An arc in a circle -/
structure Arc (c : Circle) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The length of a chord -/
def chordLength (c : Circle) (ch : Chord c) : ℝ :=
  sorry

/-- The measure of an arc -/
def arcMeasure (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- A chord subtends an arc -/
def subtends (c : Circle) (ch : Chord c) (a : Arc c) : Prop :=
  sorry

theorem equal_chords_subtend_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) :
  chordLength c ch1 = chordLength c ch2 →
  subtends c ch1 a1 →
  subtends c ch2 a2 →
  arcMeasure c a1 = arcMeasure c a2 :=
sorry

end NUMINAMATH_CALUDE_equal_chords_subtend_equal_arcs_l2797_279752


namespace NUMINAMATH_CALUDE_problem_solution_l2797_279730

def f (x : ℝ) : ℝ := |x - 5|

theorem problem_solution :
  (∃ (a b : ℝ), a = 5/2 ∧ b = 11/2 ∧
    (∀ x : ℝ, f x + f (x + 2) ≤ 3 ↔ a ≤ x ∧ x ≤ b)) ∧
  (∀ a x : ℝ, a < 0 → f (a * x) - f (5 * a) ≥ a * f x) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2797_279730


namespace NUMINAMATH_CALUDE_problem_solution_l2797_279795

theorem problem_solution :
  ∀ (a b c : ℝ),
    (∃ (x : ℝ), x > 0 ∧ (1 - 2*a)^2 = x ∧ (a + 4)^2 = x) →
    (4*a + 2*b - 1)^(1/3) = 3 →
    c = ⌊Real.sqrt 13⌋ →
    a = 5 ∧ b = 4 ∧ c = 3 ∧ Real.sqrt (a + 2*b + c) = 4 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2797_279795


namespace NUMINAMATH_CALUDE_time_after_3339_minutes_l2797_279754

/-- Represents a time of day -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat
  deriving Repr

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : TimeOfDay
  deriving Repr

def minutesToDateTime (startDateTime : DateTime) (elapsedMinutes : Nat) : DateTime :=
  sorry

theorem time_after_3339_minutes :
  let startDateTime := DateTime.mk (Date.mk 2020 12 31) (TimeOfDay.mk 0 0)
  let endDateTime := minutesToDateTime startDateTime 3339
  endDateTime = DateTime.mk (Date.mk 2021 1 2) (TimeOfDay.mk 7 39) := by
  sorry

end NUMINAMATH_CALUDE_time_after_3339_minutes_l2797_279754


namespace NUMINAMATH_CALUDE_average_multiplication_invariance_l2797_279767

theorem average_multiplication_invariance (S : Finset ℝ) (n : ℕ) (h : n > 0) :
  let avg := (S.sum id) / n
  let new_avg := (S.sum (fun x => 10 * x)) / n
  avg = 7 ∧ new_avg = 70 →
  ∃ (m : ℕ), m > 0 ∧ (S.sum id) / m = 7 ∧ (S.sum (fun x => 10 * x)) / m = 70 :=
by sorry

end NUMINAMATH_CALUDE_average_multiplication_invariance_l2797_279767


namespace NUMINAMATH_CALUDE_angle_B_in_triangle_l2797_279781

theorem angle_B_in_triangle (A B C : ℝ) (BC AC : ℝ) (h1 : BC = 6) (h2 : AC = 4) (h3 : Real.sin A = 3/4) :
  B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_in_triangle_l2797_279781


namespace NUMINAMATH_CALUDE_specific_tile_arrangement_l2797_279785

/-- The number of distinguishable arrangements for a row of tiles -/
def tileArrangements (brown purple green yellow blue : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow + blue) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green *
   Nat.factorial yellow * Nat.factorial blue)

/-- Theorem: The number of distinguishable arrangements for a row consisting of
    1 brown tile, 1 purple tile, 3 green tiles, 3 yellow tiles, and 2 blue tiles
    is equal to 50400. -/
theorem specific_tile_arrangement :
  tileArrangements 1 1 3 3 2 = 50400 := by
  sorry

end NUMINAMATH_CALUDE_specific_tile_arrangement_l2797_279785


namespace NUMINAMATH_CALUDE_triangle_problem_l2797_279764

theorem triangle_problem (A B C : Real) (a b c : Real) :
  let m : Real × Real := (Real.sqrt 3, 1 - Real.cos A)
  let n : Real × Real := (Real.sin A, -1)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a = 2) →
  (Real.cos B = Real.sqrt 3 / 3) →
  (A = 2 * Real.pi / 3 ∧ b = 4 * Real.sqrt 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2797_279764


namespace NUMINAMATH_CALUDE_gunther_free_time_l2797_279780

def cleaning_time (vacuum_time dust_time mop_time brush_time_per_cat num_cats : ℕ) : ℕ :=
  vacuum_time + dust_time + mop_time + brush_time_per_cat * num_cats

theorem gunther_free_time 
  (free_time : ℕ) 
  (vacuum_time : ℕ)
  (dust_time : ℕ)
  (mop_time : ℕ)
  (brush_time_per_cat : ℕ)
  (num_cats : ℕ)
  (h1 : free_time = 3 * 60)
  (h2 : vacuum_time = 45)
  (h3 : dust_time = 60)
  (h4 : mop_time = 30)
  (h5 : brush_time_per_cat = 5)
  (h6 : num_cats = 3) :
  free_time - cleaning_time vacuum_time dust_time mop_time brush_time_per_cat num_cats = 30 :=
by sorry

end NUMINAMATH_CALUDE_gunther_free_time_l2797_279780


namespace NUMINAMATH_CALUDE_unique_root_of_increasing_function_l2797_279782

theorem unique_root_of_increasing_function (f : ℝ → ℝ) (h : Monotone f) :
  ∃! x, f x = 0 ∨ (∀ x, f x ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_increasing_function_l2797_279782


namespace NUMINAMATH_CALUDE_largest_percentage_increase_l2797_279715

def students : Fin 6 → ℕ
  | 0 => 50  -- 2010
  | 1 => 55  -- 2011
  | 2 => 60  -- 2012
  | 3 => 72  -- 2013
  | 4 => 75  -- 2014
  | 5 => 90  -- 2015

def percentageIncrease (year : Fin 5) : ℚ :=
  (students (year.succ) - students year : ℚ) / students year * 100

theorem largest_percentage_increase :
  (∀ year : Fin 5, percentageIncrease year ≤ percentageIncrease 2 ∨ percentageIncrease year ≤ percentageIncrease 4) ∧
  percentageIncrease 2 = percentageIncrease 4 :=
sorry

end NUMINAMATH_CALUDE_largest_percentage_increase_l2797_279715


namespace NUMINAMATH_CALUDE_mckenna_start_time_l2797_279710

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Mckenna's work schedule -/
structure WorkSchedule where
  officeEndTime : Time
  meetingEndTime : Time
  workDuration : Nat
  totalWorkDuration : Nat

/-- Calculate the difference between two times in hours -/
def timeDifference (t1 t2 : Time) : Nat :=
  sorry

/-- Calculate the time after adding hours to a given time -/
def addHours (t : Time) (hours : Nat) : Time :=
  sorry

theorem mckenna_start_time (schedule : WorkSchedule)
  (h1 : schedule.officeEndTime = ⟨11, 0⟩)
  (h2 : schedule.meetingEndTime = ⟨13, 0⟩)
  (h3 : schedule.workDuration = 2)
  (h4 : schedule.totalWorkDuration = 7) :
  timeDifference ⟨8, 0⟩ (addHours schedule.meetingEndTime schedule.workDuration) = schedule.totalWorkDuration :=
sorry

end NUMINAMATH_CALUDE_mckenna_start_time_l2797_279710


namespace NUMINAMATH_CALUDE_fraction_sum_l2797_279753

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2797_279753


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2797_279757

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. --/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2797_279757


namespace NUMINAMATH_CALUDE_shopping_expenses_total_l2797_279719

/-- Represents the shopping expenses of Lisa and Carly -/
def ShoppingExpenses (lisa_tshirt : ℝ) : Prop :=
  let lisa_jeans := lisa_tshirt / 2
  let lisa_coat := lisa_tshirt * 2
  let shoe_cost := lisa_jeans * 3
  let carly_tshirt := lisa_tshirt / 4
  let carly_jeans := lisa_jeans * 3
  let carly_coat := lisa_coat / 2
  let carly_dress := shoe_cost * 2
  let lisa_total := lisa_tshirt + lisa_jeans + lisa_coat + shoe_cost
  let carly_total := carly_tshirt + carly_jeans + carly_coat + shoe_cost + carly_dress
  lisa_tshirt = 40 ∧ lisa_total + carly_total = 490

/-- Theorem stating that the total amount spent by Lisa and Carly is $490 -/
theorem shopping_expenses_total : ShoppingExpenses 40 := by
  sorry

end NUMINAMATH_CALUDE_shopping_expenses_total_l2797_279719


namespace NUMINAMATH_CALUDE_athena_total_spent_l2797_279773

def sandwich_price : ℝ := 3
def fruit_drink_price : ℝ := 2.5
def num_sandwiches : ℕ := 3
def num_fruit_drinks : ℕ := 2

theorem athena_total_spent :
  (num_sandwiches : ℝ) * sandwich_price + (num_fruit_drinks : ℝ) * fruit_drink_price = 14 := by
  sorry

end NUMINAMATH_CALUDE_athena_total_spent_l2797_279773
