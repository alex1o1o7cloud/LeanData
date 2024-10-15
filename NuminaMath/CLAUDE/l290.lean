import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l290_29094

/-- Represents a breeding room with a certain number of mice -/
structure BreedingRoom where
  mice : ℕ

/-- Represents a research institute with multiple breeding rooms -/
structure ResearchInstitute where
  rooms : List BreedingRoom

/-- Different sampling methods -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Cluster
  | Systematic

/-- Determines if a population has significant subgroup differences -/
def hasSignificantSubgroupDifferences (institute : ResearchInstitute) : Prop :=
  ∃ (r1 r2 : BreedingRoom), r1 ∈ institute.rooms ∧ r2 ∈ institute.rooms ∧ r1.mice ≠ r2.mice

/-- Determines the most appropriate sampling method given a research institute and sample size -/
def mostAppropriateSamplingMethod (institute : ResearchInstitute) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is most appropriate for the given conditions -/
theorem stratified_sampling_most_appropriate
  (institute : ResearchInstitute)
  (sampleSize : ℕ)
  (h1 : institute.rooms.length = 4)
  (h2 : institute.rooms = [⟨18⟩, ⟨24⟩, ⟨54⟩, ⟨48⟩])
  (h3 : sampleSize = 24)
  (h4 : hasSignificantSubgroupDifferences institute) :
  mostAppropriateSamplingMethod institute sampleSize = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l290_29094


namespace NUMINAMATH_CALUDE_book_price_changes_l290_29071

theorem book_price_changes (initial_price : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) (final_price : ℝ) : 
  initial_price = 400 →
  decrease_percent = 15 →
  increase_percent = 40 →
  final_price = 476 →
  initial_price * (1 - decrease_percent / 100) * (1 + increase_percent / 100) = final_price := by
sorry

end NUMINAMATH_CALUDE_book_price_changes_l290_29071


namespace NUMINAMATH_CALUDE_clock_painting_theorem_l290_29007

def clock_numbers : ℕ := 12

def paint_interval_a : ℕ := 57
def paint_interval_b : ℕ := 2005

theorem clock_painting_theorem :
  (∃ (painted_numbers : Finset ℕ),
    painted_numbers.card = 4 ∧
    ∀ n : ℕ, n ∈ painted_numbers ↔ n < clock_numbers ∧ ∃ k : ℕ, (paint_interval_a * k) % clock_numbers = n) ∧
  (∀ n : ℕ, n < clock_numbers → ∃ k : ℕ, (paint_interval_b * k) % clock_numbers = n) :=
by sorry

end NUMINAMATH_CALUDE_clock_painting_theorem_l290_29007


namespace NUMINAMATH_CALUDE_goods_train_speed_l290_29030

/-- Calculates the speed of a goods train given the conditions described in the problem -/
theorem goods_train_speed
  (passenger_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (h_passenger_speed : passenger_train_speed = 100)
  (h_goods_length : goods_train_length = 400)
  (h_passing_time : passing_time = 12) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 20 ∧
    (goods_train_speed + passenger_train_speed) * passing_time / 3.6 = goods_train_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l290_29030


namespace NUMINAMATH_CALUDE_envelope_count_l290_29026

/-- The weight of one envelope in grams -/
def envelope_weight : ℝ := 8.5

/-- The total weight of all envelopes in kilograms -/
def total_weight : ℝ := 7.48

/-- The number of envelopes sent -/
def num_envelopes : ℕ := 880

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℝ := 1000

theorem envelope_count :
  (total_weight * kg_to_g) / envelope_weight = num_envelopes := by
  sorry

end NUMINAMATH_CALUDE_envelope_count_l290_29026


namespace NUMINAMATH_CALUDE_point_inside_circle_l290_29043

theorem point_inside_circle (a b : ℝ) : 
  a ≠ b → 
  a^2 - a - Real.sqrt 2 = 0 → 
  b^2 - b - Real.sqrt 2 = 0 → 
  a^2 + b^2 < 8 := by
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l290_29043


namespace NUMINAMATH_CALUDE_intersection_of_distinct_planes_is_z_axis_l290_29019

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- A plane in cylindrical coordinates defined by a constant θ value -/
def CylindricalPlane (θ_const : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = θ_const}

/-- The z-axis in cylindrical coordinates -/
def ZAxis : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = 0}

theorem intersection_of_distinct_planes_is_z_axis (θ₁ θ₂ : ℝ) (h : θ₁ ≠ θ₂) :
  (CylindricalPlane θ₁) ∩ (CylindricalPlane θ₂) = ZAxis := by
  sorry

#check intersection_of_distinct_planes_is_z_axis

end NUMINAMATH_CALUDE_intersection_of_distinct_planes_is_z_axis_l290_29019


namespace NUMINAMATH_CALUDE_function_satisfying_equation_l290_29032

theorem function_satisfying_equation (f : ℕ → ℕ) :
  (∀ m n : ℕ, f (m * n) + f (m + n) = f m * f n + 1) →
  (∀ n : ℕ, f n = 1 ∨ f n = n + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_equation_l290_29032


namespace NUMINAMATH_CALUDE_correct_factorization_l290_29056

theorem correct_factorization (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l290_29056


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l290_29074

theorem sufficient_not_necessary :
  (∀ x : ℝ, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l290_29074


namespace NUMINAMATH_CALUDE_reseating_women_l290_29038

/-- The number of ways to reseat n women in a line, where each woman can sit in her original seat or within two positions on either side. -/
def T : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 4
  | 3 => 7
  | (n + 4) => T (n + 3) + T (n + 2) + T (n + 1)

/-- Theorem stating that the number of ways to reseat 10 women under the given conditions is 480. -/
theorem reseating_women : T 10 = 480 := by
  sorry

end NUMINAMATH_CALUDE_reseating_women_l290_29038


namespace NUMINAMATH_CALUDE_problem_solution_l290_29061

-- Define the propositions
def proposition_A (x : ℝ) : Prop := (x^2 - 4*x + 3 = 0) → (x = 3)
def proposition_B (x : ℝ) : Prop := (x > 1) → (|x| > 0)
def proposition_C (p q : Prop) : Prop := (¬p ∧ ¬q) → (¬p ∧ ¬q)
def proposition_D : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define the correctness of each statement
def statement_A_correct : Prop :=
  ∀ x : ℝ, (x ≠ 3 → x^2 - 4*x + 3 ≠ 0) ↔ proposition_A x

def statement_B_correct : Prop :=
  (∀ x : ℝ, x > 1 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ x ≤ 1)

def statement_C_incorrect : Prop :=
  ∃ p q : Prop, ¬p ∧ ¬q ∧ ¬(proposition_C p q)

def statement_D_correct : Prop :=
  (¬proposition_D) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

-- Main theorem
theorem problem_solution :
  statement_A_correct ∧ statement_B_correct ∧ statement_C_incorrect ∧ statement_D_correct :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l290_29061


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l290_29063

theorem theater_ticket_difference :
  ∀ (x y : ℕ),
    x + y = 350 →
    12 * x + 8 * y = 3320 →
    y - x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l290_29063


namespace NUMINAMATH_CALUDE_minimum_guests_l290_29073

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 406) (h2 : max_per_guest = 2.5) :
  ∃ n : ℕ, n * max_per_guest ≥ total_food ∧ ∀ m : ℕ, m * max_per_guest ≥ total_food → m ≥ n ∧ n = 163 :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l290_29073


namespace NUMINAMATH_CALUDE_cube_sum_ratio_equals_product_ratio_l290_29029

theorem cube_sum_ratio_equals_product_ratio 
  (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : d + e + f = 0) 
  (h3 : d * e * f ≠ 0) : 
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = a * b * c / (d * e * f) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ratio_equals_product_ratio_l290_29029


namespace NUMINAMATH_CALUDE_point_c_coordinates_l290_29095

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Check if a point is on a line segment -/
def isOnSegment (p q r : Point) : Prop :=
  distance p r + distance r q = distance p q

theorem point_c_coordinates :
  let a : Point := ⟨-2, 1⟩
  let b : Point := ⟨4, 9⟩
  let c : Point := ⟨22/7, 55/7⟩
  isOnSegment a c b ∧ distance a c = 4 * distance c b → c = ⟨22/7, 55/7⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l290_29095


namespace NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l290_29055

-- Define the functions
def f (x : ℝ) := abs x
def g (x : ℝ) := -x^2 - 5*x - 4

-- Define the vertical distance between the graphs
def vertical_distance (x : ℝ) := abs (f x - g x)

-- Theorem statement
theorem min_vertical_distance_is_zero :
  ∃ x : ℝ, vertical_distance x = 0 ∧ ∀ y : ℝ, vertical_distance y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l290_29055


namespace NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l290_29011

theorem no_solutions_lcm_gcd_equation :
  ¬∃ (n : ℕ), n > 0 ∧ Nat.lcm n 150 = Nat.gcd n 150 + 600 := by
sorry

end NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l290_29011


namespace NUMINAMATH_CALUDE_project_savings_percentage_l290_29069

theorem project_savings_percentage 
  (actual_investment : ℕ) 
  (savings : ℕ) 
  (h1 : actual_investment = 150000)
  (h2 : savings = 50000) :
  (savings : ℝ) / ((actual_investment : ℝ) + (savings : ℝ)) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_project_savings_percentage_l290_29069


namespace NUMINAMATH_CALUDE_A_union_B_eq_B_l290_29042

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

theorem A_union_B_eq_B : A ∪ B = B := by
  sorry

end NUMINAMATH_CALUDE_A_union_B_eq_B_l290_29042


namespace NUMINAMATH_CALUDE_unique_hyperdeficient_l290_29035

/-- Sum of all divisors of n including n itself -/
def g (n : ℕ) : ℕ := sorry

/-- A number n is hyperdeficient if g(g(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := g (g n) = n + 3

/-- There exists exactly one hyperdeficient positive integer -/
theorem unique_hyperdeficient : ∃! n : ℕ+, is_hyperdeficient n := by sorry

end NUMINAMATH_CALUDE_unique_hyperdeficient_l290_29035


namespace NUMINAMATH_CALUDE_floor_to_total_ratio_example_l290_29000

/-- The ratio of students sitting on the floor to the total number of students -/
def floor_to_total_ratio (total_students floor_students : ℕ) : ℚ :=
  floor_students / total_students

/-- Proof that the ratio of students sitting on the floor to the total number of students is 11/26 -/
theorem floor_to_total_ratio_example : 
  floor_to_total_ratio 26 11 = 11 / 26 := by
  sorry

end NUMINAMATH_CALUDE_floor_to_total_ratio_example_l290_29000


namespace NUMINAMATH_CALUDE_fraction_subtraction_theorem_l290_29028

theorem fraction_subtraction_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x := a * b / (a + b)
  let y := a * b * (b + a) / (b^2 + a*b + a^2)
  ((a - x) / (b - x) = (a / b)^2) ∧ ((a - y) / (b - y) = (a / b)^3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_theorem_l290_29028


namespace NUMINAMATH_CALUDE_total_lives_calculation_l290_29050

/-- Given 6 initial players, 9 additional players, and 5 lives per player,
    the total number of lives is 75. -/
theorem total_lives_calculation (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ)
    (h1 : initial_players = 6)
    (h2 : additional_players = 9)
    (h3 : lives_per_player = 5) :
    (initial_players + additional_players) * lives_per_player = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l290_29050


namespace NUMINAMATH_CALUDE_inequality_holds_l290_29078

theorem inequality_holds (a b c d : ℝ) : (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l290_29078


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l290_29072

/-- Represents a quilt block as described in the problem -/
structure QuiltBlock where
  size : Nat
  fully_shaded : Nat
  half_shaded : Nat
  quarter_shaded : Nat

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  (q.fully_shaded + q.half_shaded / 2 + q.quarter_shaded / 2) / (q.size * q.size)

/-- The specific quilt block described in the problem -/
def problem_quilt : QuiltBlock :=
  { size := 4
    fully_shaded := 4
    half_shaded := 8
    quarter_shaded := 4 }

theorem quilt_shaded_fraction :
  shaded_fraction problem_quilt = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l290_29072


namespace NUMINAMATH_CALUDE_die_roll_probability_l290_29096

/-- The number of sides on the die -/
def n : ℕ := 8

/-- The number of rolls -/
def r : ℕ := 12

/-- The probability of rolling a different number from the previous roll -/
def p : ℚ := (n - 1) / n

/-- The probability of rolling the same number as the previous roll -/
def q : ℚ := 1 / n

theorem die_roll_probability : 
  p^(r - 2) * q = 282475249 / 8589934592 := by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l290_29096


namespace NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l290_29031

theorem max_tickets_jane_can_buy (ticket_price : ℚ) (budget : ℚ) : 
  ticket_price = 27/2 → budget = 100 → 
  (∃ n : ℕ, n * ticket_price ≤ budget ∧ 
    ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n) → 
  (∃ n : ℕ, n * ticket_price ≤ budget ∧ 
    ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l290_29031


namespace NUMINAMATH_CALUDE_remainder_sum_l290_29054

theorem remainder_sum (x y : ℤ) 
  (hx : x ≡ 47 [ZMOD 60])
  (hy : y ≡ 26 [ZMOD 45]) :
  x + y ≡ 13 [ZMOD 15] := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l290_29054


namespace NUMINAMATH_CALUDE_ages_when_bella_turns_18_l290_29079

/-- Given the initial ages and birth years, prove the ages when Bella turns 18 -/
theorem ages_when_bella_turns_18 
  (marianne_age_2000 : ℕ)
  (bella_age_2000 : ℕ)
  (carmen_age_2000 : ℕ)
  (elli_birth_year : ℕ)
  (h1 : marianne_age_2000 = 20)
  (h2 : bella_age_2000 = 8)
  (h3 : carmen_age_2000 = 15)
  (h4 : elli_birth_year = 2003) :
  let year_bella_18 := 2000 + (18 - bella_age_2000)
  (year_bella_18 - 2000 + marianne_age_2000 = 30) ∧ 
  (year_bella_18 - 2000 + carmen_age_2000 = 33) ∧
  (year_bella_18 - elli_birth_year = 15) :=
sorry

end NUMINAMATH_CALUDE_ages_when_bella_turns_18_l290_29079


namespace NUMINAMATH_CALUDE_average_age_of_three_l290_29060

/-- The average age of three people given the average age of two of them and the age of the third -/
theorem average_age_of_three (age_a : ℝ) (age_b : ℝ) (age_c : ℝ) 
  (h1 : (age_a + age_c) / 2 = 29) 
  (h2 : age_b = 23) : 
  (age_a + age_b + age_c) / 3 = 27 := by
  sorry


end NUMINAMATH_CALUDE_average_age_of_three_l290_29060


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l290_29062

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through D(4,0)
def line_through_D (x y : ℝ) : Prop := ∃ t : ℝ, x = t*y + 4

-- Define points A and B as intersections of the line and parabola
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_D A.1 A.2 ∧ line_through_D B.1 B.2 ∧
  A ≠ B

-- State the theorem
theorem parabola_line_intersection 
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
  (∀ S : ℝ, S = (1/2) * abs (A.1 * B.2 - A.2 * B.1) → S ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l290_29062


namespace NUMINAMATH_CALUDE_bill_denomination_l290_29084

theorem bill_denomination (num_bills : ℕ) (total_value : ℕ) (denomination : ℕ) :
  num_bills = 10 →
  total_value = 50 →
  num_bills * denomination = total_value →
  denomination = 5 := by
  sorry

end NUMINAMATH_CALUDE_bill_denomination_l290_29084


namespace NUMINAMATH_CALUDE_solution_difference_l290_29053

theorem solution_difference (a b : ℝ) : 
  ((a - 4) * (a + 4) = 28 * a - 112) → 
  ((b - 4) * (b + 4) = 28 * b - 112) → 
  a ≠ b →
  a > b →
  a - b = 20 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l290_29053


namespace NUMINAMATH_CALUDE_system_solution_l290_29020

theorem system_solution : 
  ∃! (x y : ℝ), (x + 2 * Real.sqrt y = 2) ∧ (2 * Real.sqrt x + y = 2) ∧ (x = 4 - 2 * Real.sqrt 3) ∧ (y = 4 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l290_29020


namespace NUMINAMATH_CALUDE_parallel_vectors_x_coord_l290_29025

/-- Given vectors a and b in ℝ², if a + b is parallel to a - 2b, then the x-coordinate of b is 4. -/
theorem parallel_vectors_x_coord (a b : ℝ × ℝ) (h : a.1 = 2 ∧ a.2 = 1 ∧ b.2 = 2) :
  (∃ k : ℝ, (a.1 + b.1, a.2 + b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  b.1 = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_coord_l290_29025


namespace NUMINAMATH_CALUDE_expression_value_l290_29018

theorem expression_value (x y : ℝ) : 
  x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7 → 
  x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2 ∨ 
  x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l290_29018


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l290_29027

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := -25 * i / (3 + 4 * i)
  Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l290_29027


namespace NUMINAMATH_CALUDE_train_length_calculation_l290_29045

/-- Calculates the length of a train given its speed and time to cross a point. -/
def trainLength (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 18 * (1000 / 3600)) -- 18 km/h converted to m/s
  (h2 : time = 200) : 
  trainLength speed time = 1000 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l290_29045


namespace NUMINAMATH_CALUDE_game_choices_l290_29097

theorem game_choices (p : ℝ) (n : ℕ) 
  (h1 : p = 0.9375) 
  (h2 : p = 1 - 1 / n) : n = 16 := by
  sorry

end NUMINAMATH_CALUDE_game_choices_l290_29097


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l290_29041

theorem min_value_of_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094 :=
sorry

theorem equality_achieved : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l290_29041


namespace NUMINAMATH_CALUDE_parabola_vertex_l290_29013

/-- The vertex of the parabola y = 2(x-5)^2 + 3 has coordinates (5, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2*(x - 5)^2 + 3 → (5, 3) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l290_29013


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l290_29091

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem president_and_committee_selection :
  let total_people : ℕ := 10
  let committee_size : ℕ := 3
  let president_choices : ℕ := total_people
  let committee_choices : ℕ := choose (total_people - 1) committee_size
  president_choices * committee_choices = 840 :=
by sorry

end NUMINAMATH_CALUDE_president_and_committee_selection_l290_29091


namespace NUMINAMATH_CALUDE_no_repeating_stock_price_l290_29066

theorem no_repeating_stock_price (n : ℕ) : ¬ ∃ (k l : ℕ), k + l > 0 ∧ k + l ≤ 365 ∧ (1 + n / 100 : ℚ)^k * (1 - n / 100 : ℚ)^l = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_repeating_stock_price_l290_29066


namespace NUMINAMATH_CALUDE_expression_evaluation_l290_29002

theorem expression_evaluation :
  let x : ℚ := -1/3
  (-5 * x^2 + 4 + x) - 3 * (-2 * x^2 + x - 1) = 70/9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l290_29002


namespace NUMINAMATH_CALUDE_triangle_third_angle_l290_29080

theorem triangle_third_angle (a b c : ℝ) (ha : a = 40) (hb : b = 60) 
  (sum : a + b + c = 180) : c = 80 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l290_29080


namespace NUMINAMATH_CALUDE_construction_time_correct_l290_29081

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ
  delayWithoutAdditionalWorkers : ℕ

/-- The planned construction time in days -/
def plannedConstructionTime (project : ConstructionProject) : ℕ := 110

theorem construction_time_correct (project : ConstructionProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.additionalWorkers = 100)
  (h3 : project.additionalWorkersStartDay = 10)
  (h4 : project.delayWithoutAdditionalWorkers = 90) :
  plannedConstructionTime project = 110 := by
  sorry

#check construction_time_correct

end NUMINAMATH_CALUDE_construction_time_correct_l290_29081


namespace NUMINAMATH_CALUDE_quadratic_factorization_l290_29077

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 20 * y^2 - 117 * y + 72 = (C * y - 8) * (D * y - 9)) →
  C * D + C = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l290_29077


namespace NUMINAMATH_CALUDE_product_of_reals_l290_29083

theorem product_of_reals (a b : ℝ) 
  (sum_eq : a + b = 7)
  (sum_cubes_eq : a^3 + b^3 = 91) : 
  a * b = 12 := by sorry

end NUMINAMATH_CALUDE_product_of_reals_l290_29083


namespace NUMINAMATH_CALUDE_triangle_inequality_fraction_l290_29016

theorem triangle_inequality_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_fraction_l290_29016


namespace NUMINAMATH_CALUDE_guitar_price_proof_l290_29049

/-- The price Gerald paid for the guitar -/
def gerald_price : ℝ := 250

/-- The price Hendricks paid for the guitar -/
def hendricks_price : ℝ := 200

/-- The percentage discount Hendricks got compared to Gerald's price -/
def discount_percentage : ℝ := 20

theorem guitar_price_proof :
  hendricks_price = gerald_price * (1 - discount_percentage / 100) →
  gerald_price = 250 := by
  sorry

end NUMINAMATH_CALUDE_guitar_price_proof_l290_29049


namespace NUMINAMATH_CALUDE_problem_statement_l290_29010

theorem problem_statement (x : ℝ) (h : x^5 + x^4 + x = -1) :
  x^1997 + x^1998 + x^1999 + x^2000 + x^2001 + x^2002 + x^2003 + x^2004 + x^2005 + x^2006 + x^2007 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l290_29010


namespace NUMINAMATH_CALUDE_company_assets_and_price_l290_29024

theorem company_assets_and_price (A B P : ℝ) 
  (h1 : P = 1.5 * A) 
  (h2 : P = 0.8571428571428571 * (A + B)) : 
  P = 2 * B := by
sorry

end NUMINAMATH_CALUDE_company_assets_and_price_l290_29024


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l290_29023

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 17 = 0

-- Define the center and radius of C1
def center_C1 : ℝ × ℝ := (-1, -2)
def radius_C1 : ℝ := 2

-- Define the center and radius of C2
def center_C2 : ℝ × ℝ := (2, -2)
def radius_C2 : ℝ := 5

-- Define the distance between centers
def distance_between_centers : ℝ := 3

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  distance_between_centers = abs (radius_C2 - radius_C1) ∧
  distance_between_centers < radius_C1 + radius_C2 :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l290_29023


namespace NUMINAMATH_CALUDE_line_slope_and_inclination_l290_29034

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

theorem line_slope_and_inclination :
  ∃ (m θ : ℝ), 
    (∀ x y, line_equation x y → y = m * x + (1 / Real.sqrt 3)) ∧
    m = -Real.sqrt 3 / 3 ∧
    θ = 5 * π / 6 ∧
    Real.tan θ = m := by
  sorry

end NUMINAMATH_CALUDE_line_slope_and_inclination_l290_29034


namespace NUMINAMATH_CALUDE_wrong_mark_value_l290_29052

/-- Proves that the wrongly entered mark is 85 given the conditions of the problem -/
theorem wrong_mark_value (n : ℕ) (correct_mark : ℕ) (average_increase : ℚ) 
  (h1 : n = 80)
  (h2 : correct_mark = 45)
  (h3 : average_increase = 1/2) : 
  ∃ (wrong_mark : ℕ), wrong_mark = 85 ∧ 
    (wrong_mark - correct_mark : ℚ) = n * average_increase := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_value_l290_29052


namespace NUMINAMATH_CALUDE_basketball_team_combinations_l290_29033

theorem basketball_team_combinations (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 6) :
  n * Nat.choose (n - 1) (k - 1) = 5544 :=
sorry

end NUMINAMATH_CALUDE_basketball_team_combinations_l290_29033


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l290_29044

/-- Proves that the area of a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1 is equal to 588 -/
theorem inscribed_circle_rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l290_29044


namespace NUMINAMATH_CALUDE_unique_number_between_2_and_5_l290_29048

theorem unique_number_between_2_and_5 (n : ℕ) : 
  2 < n ∧ n < 5 ∧ n < 10 ∧ n < 4 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_between_2_and_5_l290_29048


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l290_29021

/-- Bakery storage problem -/
theorem bakery_storage_ratio : 
  ∀ (flour baking_soda : ℝ),
  flour / baking_soda = 10 →
  flour / (baking_soda + 60) = 8 →
  ∃ (sugar : ℝ),
  sugar = 6000 ∧
  sugar / flour = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_bakery_storage_ratio_l290_29021


namespace NUMINAMATH_CALUDE_min_sum_squares_l290_29004

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ x^2 + y^2 + z^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l290_29004


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l290_29040

theorem rectangle_perimeter_bound (a b : ℝ) (h : a > 0 ∧ b > 0) (area_gt_perimeter : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l290_29040


namespace NUMINAMATH_CALUDE_solution_mixture_problem_l290_29067

/-- Solution X and Y mixture problem -/
theorem solution_mixture_problem 
  (total : ℝ) (total_pos : 0 < total)
  (x : ℝ) (x_nonneg : 0 ≤ x) (x_le_total : x ≤ total)
  (ha : x * 0.2 + (total - x) * 0.3 = total * 0.22) :
  x / total = 0.8 := by
sorry

end NUMINAMATH_CALUDE_solution_mixture_problem_l290_29067


namespace NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l290_29015

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two 2D vectors are parallel -/
def areParallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Definition of vector a -/
def a (x : ℝ) : Vector2D :=
  ⟨2, x - 1⟩

/-- Definition of vector b -/
def b (x : ℝ) : Vector2D :=
  ⟨x + 1, 4⟩

/-- Theorem stating that x = 3 is a sufficient but not necessary condition for a ∥ b -/
theorem x_eq_3_sufficient_not_necessary :
  (∃ (x : ℝ), x ≠ 3 ∧ areParallel (a x) (b x)) ∧
  (∀ (x : ℝ), x = 3 → areParallel (a x) (b x)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l290_29015


namespace NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_9_l290_29058

theorem fourth_root_16_times_sixth_root_9 : 
  (16 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/6) = 2 * (3 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_9_l290_29058


namespace NUMINAMATH_CALUDE_age_ratio_proof_l290_29085

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 32 →  -- The total of the ages of a, b, and c is 32
  b = 12 →  -- b is 12 years old
  b = 2 * c :=  -- The ratio of b's age to c's age is 2:1
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l290_29085


namespace NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l290_29092

theorem squared_difference_of_quadratic_roots :
  ∀ d e : ℝ, (3 * d^2 + 10 * d - 25 = 0) → (3 * e^2 + 10 * e - 25 = 0) →
  (d - e)^2 = 400 / 9 := by
  sorry

end NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l290_29092


namespace NUMINAMATH_CALUDE_second_account_interest_rate_l290_29082

/-- Proves that the interest rate of the second account is 0.1 given the problem conditions -/
theorem second_account_interest_rate 
  (total_investment : ℝ)
  (first_account_rate : ℝ)
  (first_account_investment : ℝ)
  (h_total : total_investment = 7200)
  (h_first_rate : first_account_rate = 0.08)
  (h_first_inv : first_account_investment = 4000)
  (h_equal_interest : first_account_rate * first_account_investment = 
    (total_investment - first_account_investment) * (10 / 100)) :
  (10 : ℝ) / 100 = (first_account_rate * first_account_investment) / (total_investment - first_account_investment) :=
by sorry

end NUMINAMATH_CALUDE_second_account_interest_rate_l290_29082


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l290_29086

theorem cubic_roots_sum (a b c : ℂ) (r s t : ℝ) : 
  (∀ x, x^3 - 3*x^2 + 5*x + 7 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ (x = a + b ∨ x = b + c ∨ x = c + a)) →
  t = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l290_29086


namespace NUMINAMATH_CALUDE_tagalong_boxes_per_case_l290_29003

theorem tagalong_boxes_per_case 
  (total_boxes : ℕ) 
  (total_cases : ℕ) 
  (h1 : total_boxes = 36) 
  (h2 : total_cases = 3) 
  (h3 : total_cases > 0) : 
  total_boxes / total_cases = 12 := by
sorry

end NUMINAMATH_CALUDE_tagalong_boxes_per_case_l290_29003


namespace NUMINAMATH_CALUDE_representation_of_2021_l290_29075

theorem representation_of_2021 : ∃ (a b c : ℤ), 2021 = a^2 - b^2 + c^2 := by
  -- We need to prove that there exist integers a, b, and c such that
  -- 2021 = a^2 - b^2 + c^2
  sorry

end NUMINAMATH_CALUDE_representation_of_2021_l290_29075


namespace NUMINAMATH_CALUDE_five_ruble_coins_count_l290_29036

/-- Given the total number of coins and the number of coins that are not of each other denomination,
    prove that the number of five-ruble coins is 5. -/
theorem five_ruble_coins_count
  (total_coins : ℕ)
  (not_two_ruble : ℕ)
  (not_ten_ruble : ℕ)
  (not_one_ruble : ℕ)
  (h1 : total_coins = 25)
  (h2 : not_two_ruble = 19)
  (h3 : not_ten_ruble = 20)
  (h4 : not_one_ruble = 16) :
  total_coins - ((total_coins - not_two_ruble) + (total_coins - not_ten_ruble) + (total_coins - not_one_ruble)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_five_ruble_coins_count_l290_29036


namespace NUMINAMATH_CALUDE_distribution_count_correct_l290_29012

/-- The number of ways to distribute 4 distinct objects into 4 distinct containers 
    such that exactly one container contains 2 objects and the others contain 1 object each -/
def distributionCount : ℕ := 144

/-- The number of universities -/
def numUniversities : ℕ := 4

/-- The number of students -/
def numStudents : ℕ := 4

theorem distribution_count_correct :
  distributionCount = 
    (numStudents.choose 2) * (numUniversities * (numUniversities - 1) * (numUniversities - 2)) :=
by sorry

end NUMINAMATH_CALUDE_distribution_count_correct_l290_29012


namespace NUMINAMATH_CALUDE_three_heads_probability_l290_29093

/-- The probability of getting heads on a single flip of a biased coin. -/
def p_heads : ℚ := 1 / 3

/-- The number of consecutive flips we're considering. -/
def n_flips : ℕ := 3

/-- The probability of getting n_flips consecutive heads. -/
def p_all_heads : ℚ := p_heads ^ n_flips

theorem three_heads_probability :
  p_all_heads = 1 / 27 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l290_29093


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l290_29068

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^2 + a*x - 3*a < 0) → a > (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l290_29068


namespace NUMINAMATH_CALUDE_andrena_debelyn_difference_l290_29008

/-- Represents the number of dolls each person has -/
structure DollCount where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- The initial doll counts before any transfers -/
def initial_count : DollCount :=
  { debelyn := 20, christel := 24, andrena := 0 }

/-- The number of dolls transferred from Debelyn to Andrena -/
def debelyn_transfer : ℕ := 2

/-- The number of dolls transferred from Christel to Andrena -/
def christel_transfer : ℕ := 5

/-- The final doll counts after transfers -/
def final_count : DollCount :=
  { debelyn := initial_count.debelyn - debelyn_transfer,
    christel := initial_count.christel - christel_transfer,
    andrena := initial_count.andrena + debelyn_transfer + christel_transfer }

/-- Andrena has 2 more dolls than Christel after transfers -/
axiom andrena_christel_difference : final_count.andrena = final_count.christel + 2

/-- The theorem to be proved -/
theorem andrena_debelyn_difference :
  final_count.andrena - final_count.debelyn = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrena_debelyn_difference_l290_29008


namespace NUMINAMATH_CALUDE_no_nonzero_triple_sum_equals_third_l290_29047

theorem no_nonzero_triple_sum_equals_third : 
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    (a + b = c) ∧ (b + c = a) ∧ (c + a = b) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_triple_sum_equals_third_l290_29047


namespace NUMINAMATH_CALUDE_sin_pi_sixth_minus_two_alpha_l290_29037

theorem sin_pi_sixth_minus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sixth_minus_two_alpha_l290_29037


namespace NUMINAMATH_CALUDE_multiples_count_l290_29005

def count_multiples (n : ℕ) : ℕ := 
  (Finset.filter (λ x => x % 2 = 0 ∨ x % 3 = 0) (Finset.range (n + 1))).card

def count_multiples_not_five (n : ℕ) : ℕ := 
  (Finset.filter (λ x => (x % 2 = 0 ∨ x % 3 = 0) ∧ x % 5 ≠ 0) (Finset.range (n + 1))).card

theorem multiples_count : count_multiples_not_five 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_l290_29005


namespace NUMINAMATH_CALUDE_systematic_sampling_example_l290_29087

def isValidSystematicSample (n : ℕ) (k : ℕ) (sample : List ℕ) : Prop :=
  let interval := n / k
  sample.length = k ∧
  ∀ i, i ∈ sample → i < n ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → j - i = interval

theorem systematic_sampling_example :
  isValidSystematicSample 50 5 [1, 11, 21, 31, 41] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_example_l290_29087


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_4_l290_29098

theorem factorization_of_4x_squared_minus_4 (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_4_l290_29098


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l290_29070

/-- A quadratic function of the form y = (m-2)x^2 + 2x - 3 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 2 * x - 3

/-- The range of m for which the function is quadratic -/
theorem quadratic_function_m_range (m : ℝ) :
  (∃ (a : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), quadratic_function m x = a * x^2 + 2 * x - 3) ↔ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l290_29070


namespace NUMINAMATH_CALUDE_three_circles_cover_horizon_two_circles_cannot_cover_horizon_l290_29090

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

-- Define a point in 2D space
def Point : Type := ℝ × ℝ

-- Define a ray emanating from a point
structure Ray where
  origin : Point
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Function to check if two circles are non-overlapping and non-touching
def non_overlapping_non_touching (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance > c1.radius + c2.radius

-- Function to check if a point is outside a circle
def point_outside_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Function to check if a ray intersects a circle
def ray_intersects_circle (r : Ray) (c : Circle) : Prop :=
  sorry  -- The actual implementation would go here

-- Theorem for three circles covering the horizon
theorem three_circles_cover_horizon :
  ∃ (c1 c2 c3 : Circle) (p : Point),
    non_overlapping_non_touching c1 c2 ∧
    non_overlapping_non_touching c1 c3 ∧
    non_overlapping_non_touching c2 c3 ∧
    point_outside_circle p c1 ∧
    point_outside_circle p c2 ∧
    point_outside_circle p c3 ∧
    ∀ (r : Ray), r.origin = p →
      ray_intersects_circle r c1 ∨
      ray_intersects_circle r c2 ∨
      ray_intersects_circle r c3 :=
  sorry

-- Theorem for two circles not covering the horizon
theorem two_circles_cannot_cover_horizon :
  ¬ ∃ (c1 c2 : Circle) (p : Point),
    non_overlapping_non_touching c1 c2 ∧
    point_outside_circle p c1 ∧
    point_outside_circle p c2 ∧
    ∀ (r : Ray), r.origin = p →
      ray_intersects_circle r c1 ∨
      ray_intersects_circle r c2 :=
  sorry

end NUMINAMATH_CALUDE_three_circles_cover_horizon_two_circles_cannot_cover_horizon_l290_29090


namespace NUMINAMATH_CALUDE_area_circular_segment_equilateral_triangle_l290_29022

/-- The area of a circular segment cut off by one side of an inscribed equilateral triangle -/
theorem area_circular_segment_equilateral_triangle (a : ℝ) (ha : a > 0) :
  let R := a / Real.sqrt 3
  let sector_area := π * R^2 / 3
  let triangle_area := a * R / 4
  sector_area - triangle_area = (a^2 * (4 * π - 3 * Real.sqrt 3)) / 36 := by
  sorry

end NUMINAMATH_CALUDE_area_circular_segment_equilateral_triangle_l290_29022


namespace NUMINAMATH_CALUDE_binomial_variance_specific_case_l290_29014

-- Define the parameters
def n : ℕ := 10
def p : ℝ := 0.02

-- Define the variance function for a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Theorem statement
theorem binomial_variance_specific_case :
  binomial_variance n p = 0.196 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_specific_case_l290_29014


namespace NUMINAMATH_CALUDE_valid_pairs_count_l290_29089

/-- A function that counts the number of valid (a,b) pairs -/
def count_valid_pairs : ℕ :=
  (Finset.range 50).sum (fun a => 
    Nat.ceil (((a + 1) : ℕ) / 2))

/-- The main theorem stating that there are exactly 75 valid pairs -/
theorem valid_pairs_count : count_valid_pairs = 75 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l290_29089


namespace NUMINAMATH_CALUDE_bobby_candy_count_l290_29001

theorem bobby_candy_count (initial : ℕ) (additional : ℕ) : 
  initial = 26 → additional = 17 → initial + additional = 43 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l290_29001


namespace NUMINAMATH_CALUDE_vacation_emails_l290_29039

theorem vacation_emails (a₁ : ℝ) (r : ℝ) (n : ℕ) (h1 : a₁ = 16) (h2 : r = 1/2) (h3 : n = 4) :
  a₁ * (1 - r^n) / (1 - r) = 30 := by
  sorry

end NUMINAMATH_CALUDE_vacation_emails_l290_29039


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l290_29088

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n^2 < 900 → ¬(2 ∣ n^2 ∧ 3 ∣ n^2 ∧ 5 ∣ n^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l290_29088


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l290_29006

theorem quadratic_root_proof (x : ℝ) : 
  x = (-31 - Real.sqrt 481) / 12 → 6 * x^2 + 31 * x + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l290_29006


namespace NUMINAMATH_CALUDE_alex_bike_trip_l290_29076

/-- Alex's bike trip problem -/
theorem alex_bike_trip (v : ℝ) 
  (h1 : 4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) : v = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_bike_trip_l290_29076


namespace NUMINAMATH_CALUDE_man_birth_year_proof_l290_29099

theorem man_birth_year_proof : ∃! x : ℤ,
  x^2 - x = 1640 ∧
  2*(x + 2*x) = 2*x ∧
  x^2 - x < 1825 := by
  sorry

end NUMINAMATH_CALUDE_man_birth_year_proof_l290_29099


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l290_29009

theorem smallest_angle_in_triangle (d e f : ℝ) (F : ℝ) (h1 : d = 2) (h2 : e = 2) (h3 : f > 4 * Real.sqrt 2) :
  let y := Real.pi
  F ≥ y ∧ ∀ z, z < y → F > z :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l290_29009


namespace NUMINAMATH_CALUDE_valentines_day_equality_l290_29057

theorem valentines_day_equality (m d : ℕ) : 
  (∃ k : ℕ, 
    5 * m = 3 * k + 2 * (d - 3) ∧ 
    4 * d = 2 * k + 2 * (m - 2)) → 
  m = d :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_day_equality_l290_29057


namespace NUMINAMATH_CALUDE_min_value_problem_l290_29046

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l290_29046


namespace NUMINAMATH_CALUDE_unique_function_solution_l290_29059

-- Define the property that f should satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1

-- State the theorem
theorem unique_function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → (∀ x : ℝ, f x = x) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l290_29059


namespace NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l290_29065

/-- The scientific notation representation of 21 billion -/
def twenty_one_billion_scientific : ℝ := 2.1 * (10 ^ 9)

/-- The value of 21 billion -/
def twenty_one_billion : ℝ := 21 * (10 ^ 9)

theorem twenty_one_billion_scientific_notation :
  twenty_one_billion = twenty_one_billion_scientific :=
by sorry

end NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l290_29065


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l290_29017

/-- The volume of a rectangular solid with given face areas -/
theorem rectangular_solid_volume
  (side_face_area front_face_area bottom_face_area : ℝ)
  (h_side : side_face_area = 18)
  (h_front : front_face_area = 15)
  (h_bottom : bottom_face_area = 10) :
  ∃ (x y z : ℝ),
    x * y = side_face_area ∧
    y * z = front_face_area ∧
    z * x = bottom_face_area ∧
    x * y * z = 30 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l290_29017


namespace NUMINAMATH_CALUDE_card_probability_l290_29064

def cards : Finset ℕ := Finset.range 11

def group_A : Finset ℕ := cards.filter (λ x => x % 2 = 1)
def group_B : Finset ℕ := cards.filter (λ x => x % 2 = 0)

def average (a b c : ℕ) : ℚ := (a + b + c : ℚ) / 3

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (group_A.product group_B).filter (λ (a, b) => a + b < 6)

theorem card_probability :
  (favorable_outcomes.card : ℚ) / (group_A.card * group_B.card) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_card_probability_l290_29064


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l290_29051

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 3*x + 10 > 0} = Set.Ioo (-2) 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l290_29051
