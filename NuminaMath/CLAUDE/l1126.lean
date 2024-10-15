import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_through_points_l1126_112607

/-- The standard form of a hyperbola passing through two given points -/
theorem hyperbola_through_points :
  let P₁ : ℝ × ℝ := (3, -4 * Real.sqrt 2)
  let P₂ : ℝ × ℝ := (9/4, 5)
  let hyperbola (x y : ℝ) := 49 * x^2 - 7 * y^2 = 113
  (hyperbola P₁.1 P₁.2) ∧ (hyperbola P₂.1 P₂.2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_through_points_l1126_112607


namespace NUMINAMATH_CALUDE_legs_in_pool_l1126_112628

/-- The number of people in Karen and Donald's family -/
def karen_donald_family : ℕ := 8

/-- The number of people in Tom and Eva's family -/
def tom_eva_family : ℕ := 6

/-- The total number of people in both families -/
def total_people : ℕ := karen_donald_family + tom_eva_family

/-- The number of people not in the pool -/
def people_not_in_pool : ℕ := 6

/-- The number of legs per person -/
def legs_per_person : ℕ := 2

theorem legs_in_pool : 
  (total_people - people_not_in_pool) * legs_per_person = 16 := by
  sorry

end NUMINAMATH_CALUDE_legs_in_pool_l1126_112628


namespace NUMINAMATH_CALUDE_harris_dog_carrot_cost_l1126_112692

/-- The annual cost of carrots for Harris's dog -/
def annual_carrot_cost (carrots_per_day : ℕ) (carrots_per_bag : ℕ) (cost_per_bag : ℚ) (days_per_year : ℕ) : ℚ :=
  (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag

/-- Theorem stating the annual cost of carrots for Harris's dog -/
theorem harris_dog_carrot_cost :
  annual_carrot_cost 1 5 2 365 = 146 := by
  sorry

end NUMINAMATH_CALUDE_harris_dog_carrot_cost_l1126_112692


namespace NUMINAMATH_CALUDE_max_ab_value_l1126_112604

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a + 4*b + a*b = 3) :
  a * b ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1126_112604


namespace NUMINAMATH_CALUDE_line_slope_angle_l1126_112621

theorem line_slope_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 1 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l1126_112621


namespace NUMINAMATH_CALUDE_x_squared_plus_nine_x_over_x_minus_three_squared_equals_90_l1126_112602

theorem x_squared_plus_nine_x_over_x_minus_three_squared_equals_90 (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 90 →
  ((x - 3)^2 * (x + 4)) / (3 * x - 4) = 36 / 11 ∨
  ((x - 3)^2 * (x + 4)) / (3 * x - 4) = 468 / 23 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_nine_x_over_x_minus_three_squared_equals_90_l1126_112602


namespace NUMINAMATH_CALUDE_larans_weekly_profit_l1126_112669

/-- Represents Laran's poster business --/
structure PosterBusiness where
  total_posters_per_day : ℕ
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_price : ℕ
  small_poster_cost : ℕ

/-- Calculates the weekly profit for the poster business --/
def weekly_profit (business : PosterBusiness) : ℕ :=
  let small_posters_per_day := business.total_posters_per_day - business.large_posters_per_day
  let large_poster_profit := business.large_poster_price - business.large_poster_cost
  let small_poster_profit := business.small_poster_price - business.small_poster_cost
  let daily_profit := business.large_posters_per_day * large_poster_profit + small_posters_per_day * small_poster_profit
  5 * daily_profit

/-- Laran's poster business setup --/
def larans_business : PosterBusiness :=
  { total_posters_per_day := 5
  , large_posters_per_day := 2
  , large_poster_price := 10
  , large_poster_cost := 5
  , small_poster_price := 6
  , small_poster_cost := 3 }

/-- Theorem stating that Laran's weekly profit is $95 --/
theorem larans_weekly_profit :
  weekly_profit larans_business = 95 := by
  sorry


end NUMINAMATH_CALUDE_larans_weekly_profit_l1126_112669


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l1126_112687

theorem midpoint_x_coordinate_sum (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l1126_112687


namespace NUMINAMATH_CALUDE_coin_value_equality_l1126_112619

theorem coin_value_equality (n : ℕ) : 
  25 * 25 + 20 * 10 = 15 * 25 + 10 * 10 + n * 50 → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_equality_l1126_112619


namespace NUMINAMATH_CALUDE_linda_egg_ratio_l1126_112690

theorem linda_egg_ratio : 
  ∀ (total_eggs : ℕ) (brown_eggs : ℕ) (white_eggs : ℕ),
  total_eggs = 12 →
  brown_eggs = 5 →
  total_eggs = brown_eggs + white_eggs →
  (white_eggs : ℚ) / (brown_eggs : ℚ) = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_linda_egg_ratio_l1126_112690


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l1126_112680

-- Part 1
def inequality_solution_set (x : ℝ) : Prop :=
  x^2 - 5*x + 4 > 0

theorem solve_inequality :
  ∀ x : ℝ, inequality_solution_set x ↔ (x < 1 ∨ x > 4) :=
sorry

-- Part 2
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + 4 > 0

theorem range_of_a :
  ∀ a : ℝ, always_positive a ↔ -4 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l1126_112680


namespace NUMINAMATH_CALUDE_consecutive_math_majors_probability_l1126_112655

/-- The number of people sitting around the table -/
def total_people : ℕ := 11

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of all math majors sitting consecutively -/
def prob_consecutive_math_majors : ℚ := 1 / 4320

theorem consecutive_math_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := (total_people - math_majors + 1) * Nat.factorial math_majors
  Rat.cast favorable_arrangements / Rat.cast total_arrangements = prob_consecutive_math_majors := by
  sorry

end NUMINAMATH_CALUDE_consecutive_math_majors_probability_l1126_112655


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1126_112672

/-- Given a line in vector form, prove it's equivalent to the slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (-3 : ℝ) * (x - 3) + (-7 : ℝ) * (y - 14) = 0 ↔ 
  y = (-3/7 : ℝ) * x + 107/7 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1126_112672


namespace NUMINAMATH_CALUDE_unique_intersecting_line_l1126_112626

/-- A parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 8 * p.1}

/-- The point M with coordinates (2, 4) -/
def M : ℝ × ℝ := (2, 4)

/-- A line that passes through point M and intersects the parabola at exactly one point -/
def UniqueLine (l : Set (ℝ × ℝ)) : Prop :=
  M ∈ l ∧ (∃! p, p ∈ l ∩ Parabola)

/-- The theorem stating that there is exactly one unique line passing through M
    that intersects the parabola at exactly one point -/
theorem unique_intersecting_line :
  ∃! l : Set (ℝ × ℝ), UniqueLine l :=
sorry

end NUMINAMATH_CALUDE_unique_intersecting_line_l1126_112626


namespace NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l1126_112653

theorem factorial_plus_24_equals_square (n m : ℕ) : n.factorial + 24 = m ^ 2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l1126_112653


namespace NUMINAMATH_CALUDE_opposite_of_sqrt3_plus_a_l1126_112617

theorem opposite_of_sqrt3_plus_a (a b : ℝ) (h : |a - 3*b| + Real.sqrt (b + 1) = 0) :
  -(Real.sqrt 3 + a) = 3 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt3_plus_a_l1126_112617


namespace NUMINAMATH_CALUDE_composite_function_ratio_l1126_112633

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem composite_function_ratio :
  (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_ratio_l1126_112633


namespace NUMINAMATH_CALUDE_count_integers_with_same_remainder_l1126_112662

theorem count_integers_with_same_remainder : ∃! (S : Finset ℕ),
  (∀ n ∈ S, 150 < n ∧ n < 250 ∧ ∃ r : ℕ, r ≤ 6 ∧ n % 7 = r ∧ n % 9 = r) ∧
  S.card = 7 := by sorry

end NUMINAMATH_CALUDE_count_integers_with_same_remainder_l1126_112662


namespace NUMINAMATH_CALUDE_weighted_graph_vertex_labeling_l1126_112663

-- Define a graph type
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define a weight function type
def WeightFunction (V : Type) := V → V → ℝ

-- Define the property of distinct positive weights
def DistinctPositiveWeights (V : Type) (f : WeightFunction V) :=
  ∀ u v w : V, u ≠ v → v ≠ w → u ≠ w → f u v > 0 ∧ f u v ≠ f v w ∧ f u v ≠ f u w

-- Define the degenerate triangle property
def DegenerateTriangle (V : Type) (f : WeightFunction V) :=
  ∀ a b c : V, 
    (f a b = f a c + f b c) ∨ 
    (f a c = f a b + f b c) ∨ 
    (f b c = f a b + f a c)

-- Define the vertex labeling function type
def VertexLabeling (V : Type) := V → ℝ

-- State the theorem
theorem weighted_graph_vertex_labeling 
  (V : Type) 
  (G : Graph V) 
  (f : WeightFunction V) 
  (h1 : DistinctPositiveWeights V f) 
  (h2 : DegenerateTriangle V f) :
  ∃ w : VertexLabeling V, ∀ u v : V, f u v = |w u - w v| :=
sorry

end NUMINAMATH_CALUDE_weighted_graph_vertex_labeling_l1126_112663


namespace NUMINAMATH_CALUDE_sids_remaining_fraction_l1126_112667

/-- Proves that the fraction of Sid's original money left after purchases is 1/2 -/
theorem sids_remaining_fraction (initial : ℝ) (accessories : ℝ) (snacks : ℝ) (extra : ℝ) 
  (h1 : initial = 48)
  (h2 : accessories = 12)
  (h3 : snacks = 8)
  (h4 : extra = 4)
  (h5 : initial - (accessories + snacks) = initial * (1/2) + extra) :
  (initial - (accessories + snacks)) / initial = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sids_remaining_fraction_l1126_112667


namespace NUMINAMATH_CALUDE_sqrt_two_sum_l1126_112671

theorem sqrt_two_sum : 2 * Real.sqrt 2 + 3 * Real.sqrt 2 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_sum_l1126_112671


namespace NUMINAMATH_CALUDE_rachels_age_problem_l1126_112644

/-- Rachel's age problem -/
theorem rachels_age_problem (rachel_age : ℕ) (grandfather_age : ℕ) (mother_age : ℕ) (father_age : ℕ) :
  rachel_age = 12 →
  grandfather_age = 7 * rachel_age →
  father_age = mother_age + 5 →
  father_age + (25 - rachel_age) = 60 →
  mother_age / grandfather_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rachels_age_problem_l1126_112644


namespace NUMINAMATH_CALUDE_female_employees_at_least_60_l1126_112613

/-- Represents the number of employees in different categories -/
structure EmployeeCount where
  total : Nat
  advancedDegree : Nat
  collegeDegreeOnly : Nat
  maleCollegeDegreeOnly : Nat

/-- Theorem stating that the number of female employees is at least 60 -/
theorem female_employees_at_least_60 (e : EmployeeCount)
  (h1 : e.total = 200)
  (h2 : e.advancedDegree = 100)
  (h3 : e.collegeDegreeOnly = 100)
  (h4 : e.maleCollegeDegreeOnly = 40) :
  ∃ (femaleCount : Nat), femaleCount ≥ 60 ∧ femaleCount ≤ e.total :=
by sorry

end NUMINAMATH_CALUDE_female_employees_at_least_60_l1126_112613


namespace NUMINAMATH_CALUDE_jerome_toy_cars_l1126_112670

theorem jerome_toy_cars (original : ℕ) (total : ℕ) (last_month : ℕ) :
  original = 25 →
  total = 40 →
  total = original + last_month + 2 * last_month →
  last_month = 5 := by
  sorry

end NUMINAMATH_CALUDE_jerome_toy_cars_l1126_112670


namespace NUMINAMATH_CALUDE_sequence_squared_l1126_112614

theorem sequence_squared (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) > a n) 
  (h3 : ∀ n, (a (n + 1))^2 + (a n)^2 + 1 = 2 * ((a (n + 1)) * (a n) + a (n + 1) + a n)) :
  ∀ n, a n = n^2 := by sorry

end NUMINAMATH_CALUDE_sequence_squared_l1126_112614


namespace NUMINAMATH_CALUDE_quartic_equation_solutions_l1126_112624

theorem quartic_equation_solutions :
  ∀ x : ℝ, x^4 - x^2 - 2 = 0 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_solutions_l1126_112624


namespace NUMINAMATH_CALUDE_trapezium_area_l1126_112675

-- Define the trapezium properties
def a : ℝ := 10 -- Length of one parallel side
def b : ℝ := 18 -- Length of the other parallel side
def h : ℝ := 15 -- Distance between parallel sides

-- Theorem statement
theorem trapezium_area : (1/2 : ℝ) * (a + b) * h = 210 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l1126_112675


namespace NUMINAMATH_CALUDE_number_less_than_opposite_l1126_112698

theorem number_less_than_opposite (x : ℝ) : x = -x + (-4) ↔ x + 4 = -x := by sorry

end NUMINAMATH_CALUDE_number_less_than_opposite_l1126_112698


namespace NUMINAMATH_CALUDE_horizontal_distance_calculation_l1126_112656

/-- Given a vertical climb and a ratio of vertical to horizontal movement,
    calculate the horizontal distance traveled. -/
theorem horizontal_distance_calculation
  (vertical_climb : ℝ)
  (vertical_ratio : ℝ)
  (horizontal_ratio : ℝ)
  (h_positive : vertical_climb > 0)
  (h_ratio_positive : vertical_ratio > 0 ∧ horizontal_ratio > 0)
  (h_climb : vertical_climb = 1350)
  (h_ratio : vertical_ratio / horizontal_ratio = 1 / 2) :
  vertical_climb * horizontal_ratio / vertical_ratio = 2700 := by
  sorry

end NUMINAMATH_CALUDE_horizontal_distance_calculation_l1126_112656


namespace NUMINAMATH_CALUDE_equation_roots_l1126_112638

theorem equation_roots :
  let f (x : ℝ) := x^2 - 2*x - 2/x + 1/x^2 - 13
  ∃ (a b c d : ℝ),
    (a = (5 + Real.sqrt 21) / 2) ∧
    (b = (5 - Real.sqrt 21) / 2) ∧
    (c = (-3 + Real.sqrt 5) / 2) ∧
    (d = (-3 - Real.sqrt 5) / 2) ∧
    (∀ x : ℝ, x ≠ 0 → (f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1126_112638


namespace NUMINAMATH_CALUDE_wall_photo_dimensions_l1126_112611

/-- Given a rectangular paper with width 12 inches surrounded by a wall photo 2 inches wide,
    if the area of the wall photo is 96 square inches,
    then the length of the rectangular paper is 2 inches. -/
theorem wall_photo_dimensions (paper_length : ℝ) : 
  (paper_length + 4) * 16 = 96 → paper_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_wall_photo_dimensions_l1126_112611


namespace NUMINAMATH_CALUDE_train_length_l1126_112603

/-- Given a train that crosses three platforms with different lengths and times,
    this theorem proves that the length of the train is 30 meters. -/
theorem train_length (platform1_length platform2_length platform3_length : ℝ)
                     (platform1_time platform2_time platform3_time : ℝ)
                     (h1 : platform1_length = 180)
                     (h2 : platform2_length = 250)
                     (h3 : platform3_length = 320)
                     (h4 : platform1_time = 15)
                     (h5 : platform2_time = 20)
                     (h6 : platform3_time = 25) :
  ∃ (train_length : ℝ), 
    train_length = 30 ∧ 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧
    (train_length + platform2_length) / platform2_time = 
    (train_length + platform3_length) / platform3_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1126_112603


namespace NUMINAMATH_CALUDE_graph_equation_is_intersecting_lines_l1126_112651

theorem graph_equation_is_intersecting_lines :
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 3*x*y ↔ x*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_graph_equation_is_intersecting_lines_l1126_112651


namespace NUMINAMATH_CALUDE_boat_speed_l1126_112636

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream upstream : ℝ) (h1 : downstream = 11) (h2 : upstream = 5) :
  ∃ (still_speed stream_speed : ℝ),
    still_speed + stream_speed = downstream ∧
    still_speed - stream_speed = upstream ∧
    still_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1126_112636


namespace NUMINAMATH_CALUDE_max_abs_z_plus_i_l1126_112630

theorem max_abs_z_plus_i :
  ∀ (x y : ℝ), 
    x^2/4 + y^2 = 1 →
    ∀ (z : ℂ), 
      z = x + y * Complex.I →
      ∀ (w : ℂ), 
        Complex.abs w = Complex.abs (z + Complex.I) →
        Complex.abs w ≤ 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_plus_i_l1126_112630


namespace NUMINAMATH_CALUDE_inequality_proof_l1126_112688

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1126_112688


namespace NUMINAMATH_CALUDE_continuous_injective_on_irrationals_implies_injective_monotonic_l1126_112696

/-- A function is injective on irrational numbers -/
def InjectiveOnIrrationals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Irrational x → Irrational y → x ≠ y → f x ≠ f y

/-- A function is strictly monotonic -/
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y ∨ ∀ x y : ℝ, x < y → f x > f y

theorem continuous_injective_on_irrationals_implies_injective_monotonic
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_inj_irr : InjectiveOnIrrationals f) :
  Function.Injective f ∧ StrictlyMonotonic f :=
sorry

end NUMINAMATH_CALUDE_continuous_injective_on_irrationals_implies_injective_monotonic_l1126_112696


namespace NUMINAMATH_CALUDE_polynomial_value_given_condition_l1126_112686

theorem polynomial_value_given_condition (x : ℝ) : 
  3 * x^3 - x = 1 → 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_given_condition_l1126_112686


namespace NUMINAMATH_CALUDE_car_travel_time_ratio_l1126_112693

/-- Proves that the ratio of time taken at 70 km/h to the original time is 3:2 -/
theorem car_travel_time_ratio :
  let distance : ℝ := 630
  let original_time : ℝ := 6
  let new_speed : ℝ := 70
  let new_time : ℝ := distance / new_speed
  new_time / original_time = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_ratio_l1126_112693


namespace NUMINAMATH_CALUDE_shelves_used_l1126_112608

def initial_stock : ℕ := 17
def new_shipment : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem shelves_used (initial_stock new_shipment bears_per_shelf : ℕ) :
  initial_stock = 17 →
  new_shipment = 10 →
  bears_per_shelf = 9 →
  (initial_stock + new_shipment) / bears_per_shelf = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelves_used_l1126_112608


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l1126_112674

theorem ratio_of_fractions (A B : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : (2 / 3) * A = (3 / 7) * B) : 
  A / B = 9 / 14 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l1126_112674


namespace NUMINAMATH_CALUDE_existence_of_roots_part_a_non_existence_of_roots_part_b_l1126_112676

-- Part a
theorem existence_of_roots_part_a : ∃ (a b : ℤ),
  (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧
  (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

-- Part b
theorem non_existence_of_roots_part_b : ¬∃ (a b : ℤ),
  (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧
  (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_roots_part_a_non_existence_of_roots_part_b_l1126_112676


namespace NUMINAMATH_CALUDE_total_notebooks_purchased_l1126_112631

def john_purchases : List Nat := [2, 4, 6, 8, 10]
def wife_purchases : List Nat := [3, 7, 5, 9, 11]

theorem total_notebooks_purchased : 
  (List.sum john_purchases) + (List.sum wife_purchases) = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_notebooks_purchased_l1126_112631


namespace NUMINAMATH_CALUDE_roots_transformation_l1126_112685

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 5*r₁ + 12 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 5*r₂ + 12 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 5*r₃ + 12 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 45*(3*r₁) + 324 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 45*(3*r₂) + 324 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 45*(3*r₃) + 324 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l1126_112685


namespace NUMINAMATH_CALUDE_impossibleToGather_l1126_112694

/-- Represents the number of islands and ships -/
def n : ℕ := 1002

/-- Represents the position of a ship on the circular archipelago -/
def Position := Fin n

/-- Represents the fleet of ships -/
def Fleet := Multiset Position

/-- Represents a single day's movement of two ships -/
def Move := Position × Position × Position × Position

/-- Checks if all ships are gathered on a single island -/
def allGathered (fleet : Fleet) : Prop :=
  ∃ p : Position, fleet = Multiset.replicate n p

/-- Applies a move to the fleet -/
def applyMove (fleet : Fleet) (move : Move) : Fleet :=
  sorry

/-- The main theorem stating that it's impossible to gather all ships -/
theorem impossibleToGather (initialFleet : Fleet) :
  ¬∃ (moves : List Move), allGathered (moves.foldl applyMove initialFleet) :=
sorry

end NUMINAMATH_CALUDE_impossibleToGather_l1126_112694


namespace NUMINAMATH_CALUDE_halloween_trick_or_treat_l1126_112697

theorem halloween_trick_or_treat (duration : ℕ) (houses_per_hour : ℕ) (treats_per_house : ℕ) (total_treats : ℕ) :
  duration = 4 →
  houses_per_hour = 5 →
  treats_per_house = 3 →
  total_treats = 180 →
  total_treats / (duration * houses_per_hour * treats_per_house) = 3 := by
sorry


end NUMINAMATH_CALUDE_halloween_trick_or_treat_l1126_112697


namespace NUMINAMATH_CALUDE_perpendicular_to_two_planes_implies_parallel_l1126_112699

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_two_planes_implies_parallel 
  (α β : Plane) (a : Line) 
  (h_diff : α ≠ β) 
  (h_perp_α : perpendicular a α) 
  (h_perp_β : perpendicular a β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_two_planes_implies_parallel_l1126_112699


namespace NUMINAMATH_CALUDE_school_sections_l1126_112635

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / (Nat.gcd boys girls)) + (girls / (Nat.gcd boys girls)) = 29 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l1126_112635


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1126_112679

/-- Given a quadratic inequality ax^2 - bx + 2 < 0 with solution set {x | 1 < x < 2},
    prove that the sum of coefficients a + b equals 4. -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, (1 < x ∧ x < 2) ↔ (a * x^2 - b * x + 2 < 0)) → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1126_112679


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1126_112618

theorem trigonometric_identities :
  -- Part 1
  (Real.sin (76 * π / 180) * Real.cos (74 * π / 180) + Real.sin (14 * π / 180) * Real.cos (16 * π / 180) = 1/2) ∧
  -- Part 2
  ((1 - Real.tan (59 * π / 180)) * (1 - Real.tan (76 * π / 180)) = 2) ∧
  -- Part 3
  ((Real.sin (7 * π / 180) + Real.cos (15 * π / 180) * Real.sin (8 * π / 180)) / 
   (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin (8 * π / 180)) = 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1126_112618


namespace NUMINAMATH_CALUDE_inequality_proof_l1126_112622

theorem inequality_proof (x : ℝ) : 
  x ∈ Set.Icc (1/4 : ℝ) 3 → x ≠ 2 → x ≠ 0 → (x - 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1126_112622


namespace NUMINAMATH_CALUDE_product_of_four_integers_l1126_112657

theorem product_of_four_integers (A B C D : ℕ+) 
  (sum_eq : A + B + C + D = 100)
  (relation : A + 4 = B + 4 ∧ B + 4 = C + 4 ∧ C + 4 = D * 2) : 
  A * B * C * D = 351232 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_integers_l1126_112657


namespace NUMINAMATH_CALUDE_strawberry_plants_l1126_112609

theorem strawberry_plants (P : ℕ) : 
  24 * P - 4 = 500 → P = 21 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_plants_l1126_112609


namespace NUMINAMATH_CALUDE_annual_growth_rate_annual_growth_rate_proof_l1126_112612

/-- Given a monthly growth rate, calculate the annual growth rate -/
theorem annual_growth_rate (P : ℝ) : ℝ := 
  (1 + P)^11 - 1

/-- The annual growth rate is equal to (1+P)^11 - 1, where P is the monthly growth rate -/
theorem annual_growth_rate_proof (P : ℝ) : 
  annual_growth_rate P = (1 + P)^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_annual_growth_rate_proof_l1126_112612


namespace NUMINAMATH_CALUDE_original_number_l1126_112677

theorem original_number (final_number : ℝ) (increase_percentage : ℝ) 
  (h1 : final_number = 210)
  (h2 : increase_percentage = 0.40) : 
  final_number = (1 + increase_percentage) * 150 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1126_112677


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l1126_112632

/-- The area of a triangle with base 8 cm and height 10 cm is 40 square centimeters. -/
theorem triangle_area_with_given_base_and_height :
  let base : ℝ := 8
  let height : ℝ := 10
  let area : ℝ := (1 / 2) * base * height
  area = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l1126_112632


namespace NUMINAMATH_CALUDE_employee_device_distribution_l1126_112648

theorem employee_device_distribution (E : ℝ) (E_pos : E > 0) : 
  let cell_phone := (2/3 : ℝ) * E
  let pager := (2/5 : ℝ) * E
  let both := (0.4 : ℝ) * E
  let neither := E - (cell_phone + pager - both)
  neither = (1/3 : ℝ) * E := by
sorry

end NUMINAMATH_CALUDE_employee_device_distribution_l1126_112648


namespace NUMINAMATH_CALUDE_no_partition_of_integers_l1126_112649

theorem no_partition_of_integers : ¬ ∃ (A B C : Set ℤ), 
  (A ∪ B ∪ C = Set.univ) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (∀ n : ℤ, (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 2011) ∈ C) ∨
            (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 2011) ∈ B) ∨
            (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 2011) ∈ C) ∨
            (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 2011) ∈ A) ∨
            (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 2011) ∈ B) ∨
            (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 2011) ∈ A)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_partition_of_integers_l1126_112649


namespace NUMINAMATH_CALUDE_largest_k_ratio_l1126_112616

theorem largest_k_ratio (a b c d : ℕ+) (h1 : a + b = c + d) (h2 : 2 * a * b = c * d) (h3 : a ≥ b) :
  (∀ k : ℝ, (a : ℝ) / (b : ℝ) ≥ k → k ≤ 3 + 2 * Real.sqrt 2) ∧
  (a : ℝ) / (b : ℝ) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_largest_k_ratio_l1126_112616


namespace NUMINAMATH_CALUDE_abs_4x_minus_6_not_positive_l1126_112684

theorem abs_4x_minus_6_not_positive (x : ℚ) : 
  ¬(0 < |4 * x - 6|) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_4x_minus_6_not_positive_l1126_112684


namespace NUMINAMATH_CALUDE_smallest_sum_abc_l1126_112683

theorem smallest_sum_abc (a b c : ℕ+) (h : (3 : ℕ) * a.val = (4 : ℕ) * b.val ∧ (4 : ℕ) * b.val = (7 : ℕ) * c.val) : 
  (a.val + b.val + c.val : ℕ) ≥ 61 ∧ ∃ (a' b' c' : ℕ+), (3 : ℕ) * a'.val = (4 : ℕ) * b'.val ∧ (4 : ℕ) * b'.val = (7 : ℕ) * c'.val ∧ a'.val + b'.val + c'.val = 61 :=
by
  sorry

#check smallest_sum_abc

end NUMINAMATH_CALUDE_smallest_sum_abc_l1126_112683


namespace NUMINAMATH_CALUDE_parallelogram_bisector_slope_l1126_112681

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x

/-- Checks if a line cuts a parallelogram into two congruent polygons -/
def cutsParallelogramCongruently (l : Line) (p1 p2 p3 p4 : Point) : Prop :=
  sorry -- Definition of this property

/-- The main theorem -/
theorem parallelogram_bisector_slope :
  ∀ (l : Line),
    let p1 : Point := ⟨12, 50⟩
    let p2 : Point := ⟨12, 120⟩
    let p3 : Point := ⟨30, 160⟩
    let p4 : Point := ⟨30, 90⟩
    l.passesThrough p1 ∧
    l.passesThrough p2 ∧
    l.passesThrough p3 ∧
    l.passesThrough p4 ∧
    cutsParallelogramCongruently l p1 p2 p3 p4 →
    l.slope = 5 :=
by
  sorry

#check parallelogram_bisector_slope

end NUMINAMATH_CALUDE_parallelogram_bisector_slope_l1126_112681


namespace NUMINAMATH_CALUDE_ferry_time_difference_l1126_112689

/-- Represents the properties of a ferry --/
structure Ferry where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup for the two ferries --/
def ferryProblem : Prop :=
  ∃ (P Q : Ferry),
    P.speed = 6 ∧
    P.time = 3 ∧
    P.distance = P.speed * P.time ∧
    Q.distance = 3 * P.distance ∧
    Q.speed = P.speed + 3 ∧
    Q.time = Q.distance / Q.speed ∧
    Q.time - P.time = 3

/-- The main theorem to be proved --/
theorem ferry_time_difference : ferryProblem :=
sorry

end NUMINAMATH_CALUDE_ferry_time_difference_l1126_112689


namespace NUMINAMATH_CALUDE_first_tree_groups_count_l1126_112659

/-- Represents the number of years in one ring group -/
def years_per_group : ℕ := 6

/-- Represents the number of ring groups in the second tree -/
def second_tree_groups : ℕ := 40

/-- Represents the age difference between the first and second tree in years -/
def age_difference : ℕ := 180

/-- Calculates the number of ring groups in the first tree -/
def first_tree_groups : ℕ := 
  (second_tree_groups * years_per_group + age_difference) / years_per_group

theorem first_tree_groups_count : first_tree_groups = 70 := by
  sorry

end NUMINAMATH_CALUDE_first_tree_groups_count_l1126_112659


namespace NUMINAMATH_CALUDE_segment_length_theorem_solvability_condition_l1126_112637

/-- Two mutually tangent circles with radii r₁ and r₂ -/
structure TangentCircles where
  r₁ : ℝ
  r₂ : ℝ
  r₁_pos : r₁ > 0
  r₂_pos : r₂ > 0

/-- A line intersecting two circles in four points, creating three equal segments -/
structure IntersectingLine (tc : TangentCircles) where
  d : ℝ
  d_pos : d > 0
  intersects_circles : True  -- This is a placeholder for the intersection property

/-- The main theorem relating the segment length to the radii -/
theorem segment_length_theorem (tc : TangentCircles) (l : IntersectingLine tc) :
    l.d^2 = (1/12) * (14*tc.r₁*tc.r₂ - tc.r₁^2 - tc.r₂^2) := by sorry

/-- The solvability condition for the problem -/
theorem solvability_condition (tc : TangentCircles) :
    (∃ l : IntersectingLine tc, True) ↔ 
    (7 - 4*Real.sqrt 3 ≤ tc.r₁ / tc.r₂ ∧ tc.r₁ / tc.r₂ ≤ 7 + 4*Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_segment_length_theorem_solvability_condition_l1126_112637


namespace NUMINAMATH_CALUDE_students_per_section_after_changes_l1126_112645

theorem students_per_section_after_changes 
  (initial_students_per_section : ℕ)
  (new_sections : ℕ)
  (total_sections_after : ℕ)
  (new_students : ℕ)
  (h1 : initial_students_per_section = 24)
  (h2 : new_sections = 3)
  (h3 : total_sections_after = 16)
  (h4 : new_students = 24) :
  (initial_students_per_section * (total_sections_after - new_sections) + new_students) / total_sections_after = 21 :=
by sorry

end NUMINAMATH_CALUDE_students_per_section_after_changes_l1126_112645


namespace NUMINAMATH_CALUDE_x_value_proof_l1126_112665

theorem x_value_proof (x : ℚ) (h : 3/5 - 1/4 = 4/x) : x = 80/7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1126_112665


namespace NUMINAMATH_CALUDE_number_problem_l1126_112620

theorem number_problem (x : ℝ) : 2 * x - 12 = 20 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1126_112620


namespace NUMINAMATH_CALUDE_second_question_probability_l1126_112641

theorem second_question_probability 
  (p_first : ℝ) 
  (p_neither : ℝ) 
  (p_both : ℝ) 
  (h1 : p_first = 0.65)
  (h2 : p_neither = 0.20)
  (h3 : p_both = 0.40)
  : ∃ p_second : ℝ, p_second = 0.75 ∧ 
    p_first + p_second - p_both + p_neither = 1 :=
sorry

end NUMINAMATH_CALUDE_second_question_probability_l1126_112641


namespace NUMINAMATH_CALUDE_price_of_33kg_apples_l1126_112658

/-- The price of apples for a given weight, where the first 30 kg have a different price than additional kg. -/
def applePrice (l q : ℚ) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then l * weight
  else l * 30 + q * (weight - 30)

/-- Theorem stating the price of 33 kg of apples -/
theorem price_of_33kg_apples (l q : ℚ) :
  (applePrice l q 15 = 150) →
  (applePrice l q 36 = 366) →
  (applePrice l q 33 = 333) := by
  sorry

end NUMINAMATH_CALUDE_price_of_33kg_apples_l1126_112658


namespace NUMINAMATH_CALUDE_minimum_angle_after_8_steps_l1126_112664

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Vector type -/
structure Vector2D where
  x : ℕ
  y : ℕ

/-- Function to perform one step of vector replacement -/
def replaceVector (v1 v2 : Vector2D) : (Vector2D × Vector2D) :=
  if v1.x * v1.x + v1.y * v1.y ≤ v2.x * v2.x + v2.y * v2.y then
    ({ x := v1.x + v2.x, y := v1.y + v2.y }, v2)
  else
    (v1, { x := v1.x + v2.x, y := v1.y + v2.y })

/-- Function to perform n steps of vector replacement -/
def performSteps (n : ℕ) (v1 v2 : Vector2D) : (Vector2D × Vector2D) :=
  match n with
  | 0 => (v1, v2)
  | n + 1 => 
    let (newV1, newV2) := replaceVector v1 v2
    performSteps n newV1 newV2

/-- Cotangent of the angle between two vectors -/
def cotangentAngle (v1 v2 : Vector2D) : ℚ :=
  let dotProduct := v1.x * v2.x + v1.y * v2.y
  let crossProduct := v1.x * v2.y - v1.y * v2.x
  dotProduct / crossProduct

/-- Main theorem -/
theorem minimum_angle_after_8_steps : 
  let initialV1 : Vector2D := { x := 1, y := 0 }
  let initialV2 : Vector2D := { x := 0, y := 1 }
  let (finalV1, finalV2) := performSteps 8 initialV1 initialV2
  cotangentAngle finalV1 finalV2 = 987 := by sorry

end NUMINAMATH_CALUDE_minimum_angle_after_8_steps_l1126_112664


namespace NUMINAMATH_CALUDE_unique_equal_expression_l1126_112646

theorem unique_equal_expression (x : ℝ) (h : x > 0) :
  (x^(x+1) + x^(x+1) = 2*x^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ x^(2*x+2)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(2*x+2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_equal_expression_l1126_112646


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1126_112634

-- Define the custom operation ⊛
def circledAst (a b : ℕ) : ℕ := sorry

-- Properties of ⊛
axiom circledAst_self (a : ℕ) : circledAst a a = a
axiom circledAst_zero (a : ℕ) : circledAst a 0 = 2 * a
axiom circledAst_add (a b c d : ℕ) : 
  (circledAst a b) + (circledAst c d) = circledAst (a + c) (b + d)

-- Theorems to prove
theorem problem_1 : circledAst (2 + 3) (0 + 3) = 7 := by sorry

theorem problem_2 : circledAst 1024 48 = 2000 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1126_112634


namespace NUMINAMATH_CALUDE_toys_sold_second_week_is_26_l1126_112678

/-- The number of toys sold in the second week at an online toy store. -/
def toys_sold_second_week (initial_stock : ℕ) (sold_first_week : ℕ) (toys_left : ℕ) : ℕ :=
  initial_stock - sold_first_week - toys_left

/-- Theorem stating that 26 toys were sold in the second week. -/
theorem toys_sold_second_week_is_26 :
  toys_sold_second_week 83 38 19 = 26 := by
  sorry

#eval toys_sold_second_week 83 38 19

end NUMINAMATH_CALUDE_toys_sold_second_week_is_26_l1126_112678


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l1126_112640

theorem birds_and_storks_on_fence : 
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let storks : ℕ := 4
  let total_birds : ℕ := initial_birds + additional_birds
  (total_birds - storks) = 3 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l1126_112640


namespace NUMINAMATH_CALUDE_rainwater_chickens_l1126_112695

/-- Proves that Mr. Rainwater has 18 chickens given the conditions -/
theorem rainwater_chickens :
  ∀ (goats cows chickens : ℕ),
    cows = 9 →
    goats = 4 * cows →
    goats = 2 * chickens →
    chickens = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainwater_chickens_l1126_112695


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l1126_112627

theorem polygon_angle_sum (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 + (180 - 180 / n)) = 2007 → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l1126_112627


namespace NUMINAMATH_CALUDE_machine_production_in_10_seconds_l1126_112691

/-- A machine that produces items at a constant rate -/
structure Machine where
  items_per_minute : ℕ

/-- Calculate the number of items produced in a given number of seconds -/
def items_produced (m : Machine) (seconds : ℕ) : ℚ :=
  (m.items_per_minute : ℚ) * (seconds : ℚ) / 60

theorem machine_production_in_10_seconds (m : Machine) 
  (h : m.items_per_minute = 150) : 
  items_produced m 10 = 25 := by
  sorry

#eval items_produced ⟨150⟩ 10

end NUMINAMATH_CALUDE_machine_production_in_10_seconds_l1126_112691


namespace NUMINAMATH_CALUDE_average_weight_proof_l1126_112623

theorem average_weight_proof (total_boys : Nat) (group1_boys : Nat) (group2_boys : Nat)
  (group2_avg_weight : ℝ) (total_avg_weight : ℝ) (group1_avg_weight : ℝ) :
  total_boys = 30 →
  group1_boys = 22 →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.89 →
  group1_avg_weight = 50.25 →
  (group1_boys : ℝ) * group1_avg_weight + (group2_boys : ℝ) * group2_avg_weight =
    (total_boys : ℝ) * total_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_average_weight_proof_l1126_112623


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1126_112601

theorem inverse_variation_problem (y x : ℝ) (k : ℝ) (h1 : y * x^2 = k) 
  (h2 : 6 * 3^2 = k) (h3 : 2 * x^2 = k) : x = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1126_112601


namespace NUMINAMATH_CALUDE_kayak_production_sum_l1126_112650

/-- Calculates the sum of a geometric sequence -/
def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

/-- The number of kayaks built in the first month -/
def initial_kayaks : ℕ := 8

/-- The ratio of kayaks built between consecutive months -/
def kayak_ratio : ℕ := 3

/-- The number of months of kayak production -/
def num_months : ℕ := 6

theorem kayak_production_sum :
  geometric_sum initial_kayaks kayak_ratio num_months = 2912 := by
  sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l1126_112650


namespace NUMINAMATH_CALUDE_last_gift_probability_theorem_l1126_112668

/-- Represents a circular arrangement of houses -/
structure CircularArrangement where
  numHouses : ℕ
  startHouse : ℕ

/-- Probability of moving to either neighbor -/
def moveProbability : ℚ := 1/2

/-- The probability that a specific house is the last to receive a gift -/
def lastGiftProbability (ca : CircularArrangement) : ℚ :=
  1 / (ca.numHouses - 1 : ℚ)

theorem last_gift_probability_theorem (ca : CircularArrangement) 
  (h1 : ca.numHouses = 2014) 
  (h2 : ca.startHouse < ca.numHouses) 
  (h3 : moveProbability = 1/2) :
  lastGiftProbability ca = 1/2013 := by
  sorry

end NUMINAMATH_CALUDE_last_gift_probability_theorem_l1126_112668


namespace NUMINAMATH_CALUDE_complex_number_problem_l1126_112647

theorem complex_number_problem (z : ℂ) : 
  (∃ (b : ℝ), z = b * I) → 
  (∃ (c : ℝ), (z + 2)^2 + 8 * I = c * I) → 
  z = 2 * I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1126_112647


namespace NUMINAMATH_CALUDE_simplify_expression_l1126_112654

theorem simplify_expression (y : ℝ) :
  3 * y - 7 * y^2 + 4 - (5 - 3 * y + 7 * y^2) = -14 * y^2 + 6 * y - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1126_112654


namespace NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l1126_112661

theorem right_triangle_leg_divisible_by_three (a b c : ℕ) (h : a * a + b * b = c * c) :
  3 ∣ a ∨ 3 ∣ b :=
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l1126_112661


namespace NUMINAMATH_CALUDE_four_students_arrangement_l1126_112642

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange four students in a line
    with three students standing together. -/
def arrangements_with_restriction : ℕ :=
  permutations 2 * permutations 3

theorem four_students_arrangement :
  permutations 4 - arrangements_with_restriction = 12 := by
  sorry

end NUMINAMATH_CALUDE_four_students_arrangement_l1126_112642


namespace NUMINAMATH_CALUDE_age_difference_l1126_112606

theorem age_difference (A B : ℕ) : B = 34 → A + 10 = 2 * (B - 10) → A - B = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1126_112606


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1126_112643

theorem smallest_right_triangle_area :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 4 ∨ b = 4 ∨ c = 4) →
  (a = 6 ∨ b = 6 ∨ c = 6) →
  a^2 + b^2 = c^2 →
  (1/2 * a * b) ≥ 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1126_112643


namespace NUMINAMATH_CALUDE_village_population_l1126_112682

theorem village_population (population : ℕ) : 
  (96 : ℚ) / 100 * population = 23040 → population = 24000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1126_112682


namespace NUMINAMATH_CALUDE_tan_pi_36_is_root_l1126_112660

theorem tan_pi_36_is_root : 
  let f (x : ℝ) := x^3 - 3 * Real.tan (π/12) * x^2 - 3 * x + Real.tan (π/12)
  f (Real.tan (π/36)) = 0 := by sorry

end NUMINAMATH_CALUDE_tan_pi_36_is_root_l1126_112660


namespace NUMINAMATH_CALUDE_ellipse_properties_l1126_112600

noncomputable def ellipseC (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def pointA : ℝ × ℝ := (0, 1)

def arithmeticSequence (BF1 F1F2 BF2 : ℝ) : Prop :=
  2 * F1F2 = Real.sqrt 3 * (BF1 + BF2)

def lineL (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 2)}

def outsideCircle (A P Q : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) > 0

theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  pointA ∈ ellipseC a b h →
  (∀ B ∈ ellipseC a b h, ∃ F1 F2 : ℝ × ℝ,
    arithmeticSequence (dist B F1) (dist F1 F2) (dist B F2)) →
  (ellipseC a b h = ellipseC 2 1 ⟨by norm_num, by norm_num⟩) ∧
  (∀ k : ℝ, (∀ P Q : ℝ × ℝ, P ∈ ellipseC 2 1 ⟨by norm_num, by norm_num⟩ →
                            Q ∈ ellipseC 2 1 ⟨by norm_num, by norm_num⟩ →
                            P ∈ lineL k → Q ∈ lineL k → P ≠ Q →
                            outsideCircle pointA P Q) ↔
             (k < -3/10 ∨ k > 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1126_112600


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l1126_112673

theorem second_term_of_geometric_series 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : second_term = 1.875 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l1126_112673


namespace NUMINAMATH_CALUDE_distance_to_triangle_plane_l1126_112629

-- Define the sphere and points
def Sphere : Type := ℝ × ℝ × ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the center and radius of the sphere
def S : Sphere := sorry
def radius : ℝ := 25

-- Define the points on the sphere
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distances between points
def PQ : ℝ := 15
def QR : ℝ := 20
def RP : ℝ := 25

-- Define the distance function
def distance (a b : Point) : ℝ := sorry

-- Define the function to calculate the distance from a point to a plane
def distToPlane (point : Point) (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem distance_to_triangle_plane :
  distToPlane S P Q R = 25 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_triangle_plane_l1126_112629


namespace NUMINAMATH_CALUDE_f_problem_l1126_112625

noncomputable section

variable (f : ℝ → ℝ)

axiom f_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom f_domain : ∀ x, 0 < x → ∃ y, f x = y
axiom f_property : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_6 : f 6 = 1

theorem f_problem :
  (f 1 = 0) ∧
  (∀ x, 0 < x → (f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < (-3 + 3 * Real.sqrt 17) / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_problem_l1126_112625


namespace NUMINAMATH_CALUDE_milan_phone_rate_l1126_112652

/-- Calculates the rate per minute for a phone service given the total bill, monthly fee, and minutes used. -/
def rate_per_minute (total_bill : ℚ) (monthly_fee : ℚ) (minutes : ℕ) : ℚ :=
  (total_bill - monthly_fee) / minutes

/-- Proves that given the specific conditions, the rate per minute is $0.12 -/
theorem milan_phone_rate :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let minutes : ℕ := 178
  rate_per_minute total_bill monthly_fee minutes = 0.12 := by
sorry


end NUMINAMATH_CALUDE_milan_phone_rate_l1126_112652


namespace NUMINAMATH_CALUDE_base6_product_132_14_l1126_112610

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Multiplies two base 6 numbers --/
def multiplyBase6 (a b : List Nat) : List Nat :=
  base10ToBase6 (base6ToBase10 a * base6ToBase10 b)

theorem base6_product_132_14 :
  multiplyBase6 [2, 3, 1] [4, 1] = [2, 3, 3, 2] := by sorry

end NUMINAMATH_CALUDE_base6_product_132_14_l1126_112610


namespace NUMINAMATH_CALUDE_negation_of_all_students_punctual_l1126_112666

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a student and being punctual
variable (student : U → Prop)
variable (punctual : U → Prop)

-- State the theorem
theorem negation_of_all_students_punctual :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_students_punctual_l1126_112666


namespace NUMINAMATH_CALUDE_function_inequality_l1126_112639

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x ∈ (Set.Ioo 0 (π/2)), f' x = deriv f x) →
  (∀ x ∈ (Set.Ioo 0 (π/2)), f x - f' x * (tan x) < 0) →
  (f 1 / sin 1 > 2 * f (π/6)) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l1126_112639


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l1126_112605

def num_vegetables : ℕ := 4
def num_plots : ℕ := 3
def num_to_select : ℕ := 3

theorem vegetable_planting_methods :
  (num_vegetables - 1).choose (num_to_select - 1) * num_to_select.factorial = 18 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l1126_112605


namespace NUMINAMATH_CALUDE_min_turns_10x10_grid_l1126_112615

/-- Represents a city grid -/
structure CityGrid where
  parallel_streets : ℕ
  intersecting_streets : ℕ

/-- Represents a bus route in the city -/
structure BusRoute where
  turns : ℕ
  closed : Bool
  covers_all_intersections : Bool

/-- The minimum number of turns for a valid bus route -/
def min_turns (city : CityGrid) : ℕ := 2 * (city.parallel_streets + city.intersecting_streets)

/-- Theorem stating the minimum number of turns for a 10x10 grid city -/
theorem min_turns_10x10_grid :
  let city : CityGrid := ⟨10, 10⟩
  let route : BusRoute := ⟨min_turns city, true, true⟩
  route.turns = 20 ∧
  ∀ (other_route : BusRoute),
    (other_route.closed ∧ other_route.covers_all_intersections) →
    other_route.turns ≥ route.turns :=
by sorry

end NUMINAMATH_CALUDE_min_turns_10x10_grid_l1126_112615
