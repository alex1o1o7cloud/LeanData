import Mathlib

namespace NUMINAMATH_CALUDE_equation_a_solution_l4081_408130

theorem equation_a_solution (x : ℝ) : 
  1/(x-1) + 3/(x-3) - 9/(x-5) + 5/(x-7) = 0 ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_a_solution_l4081_408130


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l4081_408126

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l4081_408126


namespace NUMINAMATH_CALUDE_joes_journey_time_l4081_408149

/-- Represents Joe's journey from home to school with a detour -/
def joes_journey (d : ℝ) : Prop :=
  let walking_speed : ℝ := d / 3 / 9  -- Speed to walk 1/3 of d in 9 minutes
  let running_speed : ℝ := 4 * walking_speed
  let total_walking_distance : ℝ := 2 * d / 3
  let total_running_distance : ℝ := 2 * d / 3
  let total_walking_time : ℝ := total_walking_distance / walking_speed
  let total_running_time : ℝ := total_running_distance / running_speed
  total_walking_time + total_running_time = 40.5

/-- Theorem stating that Joe's journey takes 40.5 minutes -/
theorem joes_journey_time :
  ∃ d : ℝ, d > 0 ∧ joes_journey d :=
sorry

end NUMINAMATH_CALUDE_joes_journey_time_l4081_408149


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l4081_408101

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ := sorry

-- Define a function to convert ternary to decimal
def ternary_to_decimal (t : List ℕ) : ℕ := sorry

-- Theorem statement
theorem product_of_binary_and_ternary :
  let binary_num := [true, true, false, true]  -- Represents 1101₂
  let ternary_num := [2, 0, 2]  -- Represents 202₃
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 260 := by sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l4081_408101


namespace NUMINAMATH_CALUDE_original_cyclists_l4081_408114

theorem original_cyclists (total_bill : ℕ) (left_cyclists : ℕ) (extra_payment : ℕ) :
  total_bill = 80 ∧ left_cyclists = 2 ∧ extra_payment = 2 →
  ∃ x : ℕ, x > 0 ∧ (total_bill / (x - left_cyclists) = total_bill / x + extra_payment) ∧ x = 10 :=
by
  sorry

#check original_cyclists

end NUMINAMATH_CALUDE_original_cyclists_l4081_408114


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l4081_408100

theorem min_shift_for_symmetry (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = Real.sqrt 3 * Real.cos x + Real.sin x) →
  m > 0 →
  (∀ x, f (x + m) = f (-x + m)) →
  m ≥ π / 6 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l4081_408100


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l4081_408138

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∀ k : ℕ, k < 626 → Real.sqrt k - Real.sqrt (k - 1) ≥ 0.02 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l4081_408138


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achievable_l4081_408178

theorem min_value_quadratic (x y : ℝ) : x^2 + y^2 - 8*x + 6*y + x*y + 20 ≥ -88/3 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + y^2 - 8*x + 6*y + x*y + 20 = -88/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achievable_l4081_408178


namespace NUMINAMATH_CALUDE_present_value_exponent_l4081_408191

theorem present_value_exponent 
  (Q r j m n : ℝ) 
  (hQ : Q > 0) 
  (hr : r > 0) 
  (hjm : j + m > -1) 
  (heq : Q = r / (1 + j + m) ^ n) : 
  n = Real.log (r / Q) / Real.log (1 + j + m) := by
sorry

end NUMINAMATH_CALUDE_present_value_exponent_l4081_408191


namespace NUMINAMATH_CALUDE_combined_salaries_l4081_408112

theorem combined_salaries (average_salary : ℕ) (b_salary : ℕ) (total_people : ℕ) :
  average_salary = 8200 →
  b_salary = 5000 →
  total_people = 5 →
  (average_salary * total_people) - b_salary = 36000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l4081_408112


namespace NUMINAMATH_CALUDE_expression_evaluation_l4081_408194

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (c^c - c*(c-1)^(c-1))^(c-1) = 3241792 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4081_408194


namespace NUMINAMATH_CALUDE_periodic_functions_exist_l4081_408183

-- Define a type for periodic functions
def PeriodicFunction (p : ℝ) := { f : ℝ → ℝ // ∀ x, f (x + p) = f x }

-- Define a predicate for the smallest positive period
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) :=
  (∀ x, f (x + p) = f x) ∧ (∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x)

-- Main theorem
theorem periodic_functions_exist (p₁ p₂ : ℝ) (hp₁ : 0 < p₁) (hp₂ : 0 < p₂) :
  ∃ (f₁ f₂ : ℝ → ℝ),
    SmallestPositivePeriod f₁ p₁ ∧
    SmallestPositivePeriod f₂ p₂ ∧
    ∃ (p : ℝ), ∀ x, (f₁ - f₂) (x + p) = (f₁ - f₂) x :=
by
  sorry


end NUMINAMATH_CALUDE_periodic_functions_exist_l4081_408183


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_right_l4081_408152

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  h : R = (a * Real.sqrt (b * c)) / (b + c)

/-- The angles of a triangle -/
structure Angles where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: If a triangle's circumradius satisfies the given equation, 
    then it is an isosceles right triangle -/
theorem triangle_is_isosceles_right (t : Triangle) : 
  ∃ (angles : Angles), 
    angles.α = 90 ∧ 
    angles.β = 45 ∧ 
    angles.γ = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_right_l4081_408152


namespace NUMINAMATH_CALUDE_total_snowman_drawings_l4081_408162

/-- The number of cards Melody made -/
def num_cards : ℕ := 13

/-- The number of snowman drawings on each card -/
def drawings_per_card : ℕ := 4

/-- The total number of snowman drawings printed -/
def total_drawings : ℕ := num_cards * drawings_per_card

theorem total_snowman_drawings : total_drawings = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_snowman_drawings_l4081_408162


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l4081_408198

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l4081_408198


namespace NUMINAMATH_CALUDE_baking_scoops_l4081_408187

/-- Calculates the number of scoops needed given the amount of ingredient in cups and the size of the scoop -/
def scoops_needed (cups : ℚ) (scoop_size : ℚ) : ℕ :=
  (cups / scoop_size).ceil.toNat

/-- The total number of scoops needed for flour and sugar -/
def total_scoops : ℕ :=
  scoops_needed 3 (1/3) + scoops_needed 2 (1/3)

theorem baking_scoops : total_scoops = 15 := by
  sorry

end NUMINAMATH_CALUDE_baking_scoops_l4081_408187


namespace NUMINAMATH_CALUDE_volume_ratio_theorem_l4081_408160

/-- A right rectangular prism with edge lengths -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The set of points within distance r from any point in the prism -/
def S (B : RectangularPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The volume of S(r) -/
def volume_S (B : RectangularPrism) (r : ℝ) : ℝ :=
  sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem volume_ratio_theorem (B : RectangularPrism) (coeff : VolumeCoefficients) :
    B.length = 2 ∧ B.width = 4 ∧ B.height = 6 →
    (∀ r : ℝ, volume_S B r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
    coeff.b * coeff.c / (coeff.a * coeff.d) = 66 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_theorem_l4081_408160


namespace NUMINAMATH_CALUDE_correct_amount_given_to_john_l4081_408170

/-- The amount given to John after one month -/
def amount_given_to_john (held_commission : ℕ) (advance_fees : ℕ) (incentive : ℕ) : ℕ :=
  (held_commission - advance_fees) + incentive

/-- Theorem stating the correct amount given to John -/
theorem correct_amount_given_to_john :
  amount_given_to_john 25000 8280 1780 = 18500 := by
  sorry

end NUMINAMATH_CALUDE_correct_amount_given_to_john_l4081_408170


namespace NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l4081_408120

theorem flower_shop_carnation_percentage :
  let c : ℝ := 1  -- number of carnations (arbitrary non-zero value)
  let v : ℝ := c / 3  -- number of violets
  let t : ℝ := v / 4  -- number of tulips
  let r : ℝ := t  -- number of roses
  let total : ℝ := c + v + t + r  -- total number of flowers
  (c / total) * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l4081_408120


namespace NUMINAMATH_CALUDE_f_properties_l4081_408180

def f (x : ℝ) := x^3 - 3*x^2 + 6

theorem f_properties :
  (∃ (a : ℝ), IsLocalMin f a ∧ f a = 2) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 6) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 6) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, 2 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 2) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l4081_408180


namespace NUMINAMATH_CALUDE_intersection_M_N_l4081_408133

def M : Set ℕ := {0, 2, 3, 4}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4081_408133


namespace NUMINAMATH_CALUDE_park_diagonal_ratio_l4081_408146

theorem park_diagonal_ratio :
  ∀ (long_side : ℝ) (short_side : ℝ) (diagonal : ℝ),
    short_side = long_side / 2 →
    long_side + short_side - diagonal = long_side / 3 →
    long_side / diagonal = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_park_diagonal_ratio_l4081_408146


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l4081_408129

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h1 : ∀ n : ℕ, 2 * S n = n * a n)
    (h2 : a 2 = 1) :
  (∀ n : ℕ, n ≥ 1 → a n = n - 1) ∧ is_arithmetic_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l4081_408129


namespace NUMINAMATH_CALUDE_cube_surface_area_l4081_408168

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- The vertices of the cube -/
def P : Point3D := ⟨6, 11, 11⟩
def Q : Point3D := ⟨7, 7, 2⟩
def R : Point3D := ⟨10, 2, 10⟩

theorem cube_surface_area : 
  squaredDistance P Q = squaredDistance P R ∧ 
  squaredDistance P R = squaredDistance Q R ∧
  squaredDistance Q R = 98 →
  (6 * ((squaredDistance P Q).sqrt / Real.sqrt 2)^2 : ℝ) = 294 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l4081_408168


namespace NUMINAMATH_CALUDE_square_minus_self_sum_l4081_408174

theorem square_minus_self_sum : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_sum_l4081_408174


namespace NUMINAMATH_CALUDE_third_shiny_penny_probability_l4081_408119

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5

def probability_more_than_five_draws : ℚ :=
  37 / 63

theorem third_shiny_penny_probability :
  probability_more_than_five_draws =
    (Nat.choose 5 2 * Nat.choose 4 1 +
     Nat.choose 5 1 * Nat.choose 4 2 +
     Nat.choose 5 0 * Nat.choose 4 3) /
    Nat.choose total_pennies shiny_pennies :=
by sorry

end NUMINAMATH_CALUDE_third_shiny_penny_probability_l4081_408119


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4081_408161

theorem quadratic_equation_roots (a : ℝ) (m : ℝ) :
  let x₁ : ℝ := Real.sqrt (a + 2) - Real.sqrt (8 - a) + Real.sqrt (-a^2)
  (∃ x₂ : ℝ, (1/2) * m * x₁^2 + Real.sqrt 2 * x₁ + m^2 = 0 ∧
             (1/2) * m * x₂^2 + Real.sqrt 2 * x₂ + m^2 = 0) →
  (m = 1 ∧ x₁ = -Real.sqrt 2 ∧ x₂ = -Real.sqrt 2) ∨
  (m = -2 ∧ x₁ = -Real.sqrt 2 ∧ x₂ = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4081_408161


namespace NUMINAMATH_CALUDE_rectangle_area_measurement_error_l4081_408132

theorem rectangle_area_measurement_error 
  (L W : ℝ) (L_measured W_measured : ℝ) 
  (h1 : L_measured = L * 1.2) 
  (h2 : W_measured = W * 0.9) : 
  (L_measured * W_measured - L * W) / (L * W) * 100 = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_measurement_error_l4081_408132


namespace NUMINAMATH_CALUDE_count_special_integers_l4081_408121

def f (n : ℕ) : ℚ := (n^2 + n) / 2

def is_product_of_two_primes (q : ℚ) : Prop :=
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ q = p1 * p2

theorem count_special_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, f n ≤ 1000 ∧ is_product_of_two_primes (f n)) ∧
                   (∀ n : ℕ, f n ≤ 1000 ∧ is_product_of_two_primes (f n) → n ∈ S) ∧
                   S.card = 5) :=
sorry

end NUMINAMATH_CALUDE_count_special_integers_l4081_408121


namespace NUMINAMATH_CALUDE_squash_players_l4081_408107

/-- Given a class of children with information about their sport participation,
    calculate the number of children who play squash. -/
theorem squash_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) :
  total = 38 →
  tennis = 19 →
  neither = 10 →
  both = 12 →
  ∃ (squash : ℕ), squash = 21 ∧ 
    squash = total - neither - (tennis - both) := by
  sorry

#check squash_players

end NUMINAMATH_CALUDE_squash_players_l4081_408107


namespace NUMINAMATH_CALUDE_minimal_blue_chips_l4081_408136

theorem minimal_blue_chips (r g b : ℕ) : 
  b ≥ r / 3 →
  b ≤ g / 4 →
  r + g ≥ 75 →
  (∀ b' : ℕ, b' ≥ r / 3 → b' ≤ g / 4 → b' ≥ b) →
  b = 11 := by
  sorry

end NUMINAMATH_CALUDE_minimal_blue_chips_l4081_408136


namespace NUMINAMATH_CALUDE_two_statements_true_l4081_408141

open Real

-- Define the function f
noncomputable def f (x : ℝ) := 2 * sin x * cos (abs x)

-- Define the sequence a_n
def a (n : ℕ) (k : ℝ) := n^2 + k*n + 2

theorem two_statements_true :
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∃ w : ℝ, w > 0 ∧ w = 1 ∧ ∀ x, f (x + w) = f x) ∧
  (∀ k, (∀ n : ℕ, n > 0 → a (n+1) k > a n k) → k > -3) :=
by sorry

end NUMINAMATH_CALUDE_two_statements_true_l4081_408141


namespace NUMINAMATH_CALUDE_x_range_equivalence_l4081_408109

theorem x_range_equivalence (x : ℝ) : 
  (∀ a b : ℝ, a > 0 → b > 0 → x^2 + 2*x < a/b + 16*b/a) ↔ x > -4 ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_x_range_equivalence_l4081_408109


namespace NUMINAMATH_CALUDE_students_favor_both_issues_l4081_408143

theorem students_favor_both_issues (total : ℕ) (favor_first : ℕ) (favor_second : ℕ) (against_both : ℕ)
  (h1 : total = 500)
  (h2 : favor_first = 375)
  (h3 : favor_second = 275)
  (h4 : against_both = 40) :
  total - against_both = favor_first + favor_second - 190 := by
  sorry

end NUMINAMATH_CALUDE_students_favor_both_issues_l4081_408143


namespace NUMINAMATH_CALUDE_f_five_equals_142_l4081_408177

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_five_equals_142 :
  ∃ y : ℝ, (f 2 y = 100) ∧ (f 5 y = 142) := by
  sorry

end NUMINAMATH_CALUDE_f_five_equals_142_l4081_408177


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_l4081_408190

theorem other_solution_of_quadratic (x₁ : ℚ) :
  x₁ = 3/5 →
  (30 * x₁^2 + 13 = 47 * x₁ - 2) →
  ∃ x₂ : ℚ, x₂ ≠ x₁ ∧ x₂ = 5/6 ∧ 30 * x₂^2 + 13 = 47 * x₂ - 2 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_l4081_408190


namespace NUMINAMATH_CALUDE_f_strictly_increasing_and_symmetric_l4081_408158

def f (x : ℝ) : ℝ := x^(1/3)

theorem f_strictly_increasing_and_symmetric :
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_and_symmetric_l4081_408158


namespace NUMINAMATH_CALUDE_empty_seats_count_l4081_408186

structure Section where
  capacity : Nat
  attendance : Nat

def theater : List Section := [
  { capacity := 250, attendance := 195 },
  { capacity := 180, attendance := 143 },
  { capacity := 150, attendance := 110 },
  { capacity := 300, attendance := 261 },
  { capacity := 230, attendance := 157 },
  { capacity := 90, attendance := 66 }
]

def totalCapacity : Nat := List.foldl (fun acc s => acc + s.capacity) 0 theater
def totalAttendance : Nat := List.foldl (fun acc s => acc + s.attendance) 0 theater

theorem empty_seats_count :
  totalCapacity - totalAttendance = 268 :=
by sorry

end NUMINAMATH_CALUDE_empty_seats_count_l4081_408186


namespace NUMINAMATH_CALUDE_calculation_proof_l4081_408179

theorem calculation_proof : ((0.15 * 320 + 0.12 * 480) / (2/5)) * (3/4) = 198 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4081_408179


namespace NUMINAMATH_CALUDE_distance_between_points_l4081_408127

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (7, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l4081_408127


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l4081_408197

theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) : 
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l4081_408197


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l4081_408135

theorem smaller_integer_problem (x y : ℤ) : y = 5 * x + 2 ∧ y - x = 26 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l4081_408135


namespace NUMINAMATH_CALUDE_sugar_theorem_l4081_408157

def sugar_problem (initial : ℝ) (day1_use day1_borrow : ℝ)
  (day2_buy day2_use day2_receive : ℝ)
  (day3_buy day3_use day3_return day3_borrow : ℝ)
  (day4_use day4_receive : ℝ)
  (day5_use day5_borrow day5_return : ℝ) : Prop :=
  let day1 := initial - day1_use - day1_borrow
  let day2 := day1 + day2_buy - day2_use + day2_receive
  let day3 := day2 + day3_buy - day3_use + day3_return - day3_borrow
  let day4 := day3 - day4_use + day4_receive
  let day5 := day4 - day5_use - day5_borrow + day5_return
  day5 = 63.3

theorem sugar_theorem : sugar_problem 65 18.5 5.3 30.2 12.7 4.75 20.5 8.25 2.8 1.2 9.5 6.35 10.75 3.1 3 := by
  sorry

end NUMINAMATH_CALUDE_sugar_theorem_l4081_408157


namespace NUMINAMATH_CALUDE_amys_chicken_soup_cans_l4081_408110

/-- Amy's soup purchase problem -/
theorem amys_chicken_soup_cans (total_soups : ℕ) (tomato_soup_cans : ℕ) (chicken_soup_cans : ℕ) :
  total_soups = 9 →
  tomato_soup_cans = 3 →
  total_soups = tomato_soup_cans + chicken_soup_cans →
  chicken_soup_cans = 6 := by
  sorry

end NUMINAMATH_CALUDE_amys_chicken_soup_cans_l4081_408110


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l4081_408142

/-- A convex quadrilateral with angles α, β, γ, δ -/
structure ConvexQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ

/-- Definition of a parallelogram -/
def IsParallelogram (q : ConvexQuadrilateral) : Prop :=
  q.α + q.γ = 180 ∧ q.β + q.δ = 180

/-- Definition of a trapezoid -/
def IsTrapezoid (q : ConvexQuadrilateral) : Prop :=
  q.α + q.β = 180 ∨ q.β + q.γ = 180 ∨ q.γ + q.δ = 180 ∨ q.δ + q.α = 180

theorem quadrilateral_with_equal_sine_sums (q : ConvexQuadrilateral) 
  (h : Real.sin q.α + Real.sin q.γ = Real.sin q.β + Real.sin q.δ) :
  IsParallelogram q ∨ IsTrapezoid q := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l4081_408142


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4081_408106

theorem quadratic_inequality_solution (c : ℝ) :
  (∀ x : ℝ, x^2 + 5*x - 2*c ≤ 0 ↔ -6 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4081_408106


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l4081_408103

theorem trigonometric_inequality (φ : Real) (h : 0 < φ ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l4081_408103


namespace NUMINAMATH_CALUDE_tan_2x_value_l4081_408148

theorem tan_2x_value (f : ℝ → ℝ) (x : ℝ) :
  f x = Real.sin x + Real.cos x →
  (deriv f) x = 3 * f x →
  Real.tan (2 * x) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_2x_value_l4081_408148


namespace NUMINAMATH_CALUDE_midpoint_property_l4081_408102

/-- Given two points A and B in a 2D plane, proves that if C is the midpoint of AB,
    then 3 times the x-coordinate of C minus 2 times the y-coordinate of C equals 14. -/
theorem midpoint_property (A B C : ℝ × ℝ) : 
  A = (12, 9) → B = (4, 1) → C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  3 * C.1 - 2 * C.2 = 14 := by
  sorry

#check midpoint_property

end NUMINAMATH_CALUDE_midpoint_property_l4081_408102


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l4081_408173

-- Define the set of numbers
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define the event A: "The product of the two chosen numbers is even"
def event_A (x y : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Even (x * y)

-- Define the event B: "Both chosen numbers are even"
def event_B (x y : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Even x ∧ Even y

-- Define the probability of choosing two different numbers from S
def prob_total : ℚ := 10 / 1

-- Define the probability of event A
def prob_A : ℚ := 7 / 10

-- Define the probability of event A ∩ B
def prob_A_and_B : ℚ := 1 / 10

-- Theorem statement
theorem conditional_probability_B_given_A :
  prob_A_and_B / prob_A = 1 / 7 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l4081_408173


namespace NUMINAMATH_CALUDE_inequalities_proof_l4081_408115

theorem inequalities_proof (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (a^2 > b^2) ∧ (1/a > 1/b) ∧ (a^2*b < b^3) ∧ ¬(∀ a b, a > 0 → 0 > b → a + b > 0 → a^3 < a*b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l4081_408115


namespace NUMINAMATH_CALUDE_sum_of_digits_2000_l4081_408111

/-- The number of digits in a positive integer n -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the number of digits in 2^2000 and 5^2000 is 2001 -/
theorem sum_of_digits_2000 : num_digits (2^2000) + num_digits (5^2000) = 2001 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2000_l4081_408111


namespace NUMINAMATH_CALUDE_ab_not_necessary_nor_sufficient_for_a_plus_b_l4081_408151

theorem ab_not_necessary_nor_sufficient_for_a_plus_b :
  ∃ (a b : ℝ), (a * b > 0 ∧ a + b ≤ 0) ∧
  ∃ (c d : ℝ), (c * d ≤ 0 ∧ c + d > 0) := by
  sorry

end NUMINAMATH_CALUDE_ab_not_necessary_nor_sufficient_for_a_plus_b_l4081_408151


namespace NUMINAMATH_CALUDE_oil_depth_in_horizontal_cylindrical_tank_l4081_408163

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Represents the oil in the tank --/
structure Oil where
  surfaceArea : ℝ

/-- Calculates the possible depths of oil in the tank --/
def oilDepths (tank : HorizontalCylindricalTank) (oil : Oil) : Set ℝ :=
  { h : ℝ | h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 }

/-- Theorem statement --/
theorem oil_depth_in_horizontal_cylindrical_tank
  (tank : HorizontalCylindricalTank)
  (oil : Oil)
  (h_length : tank.length = 8)
  (h_diameter : tank.diameter = 4)
  (h_surface_area : oil.surfaceArea = 16) :
  ∀ h ∈ oilDepths tank oil, h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

#check oil_depth_in_horizontal_cylindrical_tank

end NUMINAMATH_CALUDE_oil_depth_in_horizontal_cylindrical_tank_l4081_408163


namespace NUMINAMATH_CALUDE_fraction_sum_minus_eight_l4081_408156

theorem fraction_sum_minus_eight : 
  (7 : ℚ) / 3 + 11 / 5 + 19 / 9 + 37 / 17 - 8 = 628 / 765 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_minus_eight_l4081_408156


namespace NUMINAMATH_CALUDE_primer_cost_calculation_l4081_408105

/-- Represents the cost of primer per gallon before discount -/
def primer_cost : ℝ := 30

/-- Number of rooms to be painted and primed -/
def num_rooms : ℕ := 5

/-- Cost of paint per gallon -/
def paint_cost : ℝ := 25

/-- Discount rate on primer -/
def primer_discount : ℝ := 0.2

/-- Total amount spent on paint and primer -/
def total_spent : ℝ := 245

theorem primer_cost_calculation : 
  (num_rooms : ℝ) * paint_cost + 
  (num_rooms : ℝ) * primer_cost * (1 - primer_discount) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_primer_cost_calculation_l4081_408105


namespace NUMINAMATH_CALUDE_jordans_garden_area_l4081_408172

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the garden in square yards --/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * ((g.long_side_posts - 1) * g.post_spacing)

/-- Theorem stating the area of Jordan's garden --/
theorem jordans_garden_area :
  ∀ g : Garden,
    g.total_posts = 28 →
    g.post_spacing = 3 →
    g.long_side_posts = 2 * g.short_side_posts + 3 →
    garden_area g = 630 := by
  sorry

end NUMINAMATH_CALUDE_jordans_garden_area_l4081_408172


namespace NUMINAMATH_CALUDE_equation_solution_l4081_408199

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 2 → (x - 8 / (x - 2) = 5 + 8 / (x - 2)) ↔ (x = 9 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4081_408199


namespace NUMINAMATH_CALUDE_wedding_cost_theorem_l4081_408125

/-- Calculate the total cost of John's wedding based on given parameters. -/
def wedding_cost (venue_cost : ℕ) (cost_per_guest : ℕ) (johns_guests : ℕ) (wife_increase_percent : ℕ) : ℕ :=
  let total_guests := johns_guests + (johns_guests * wife_increase_percent) / 100
  venue_cost + total_guests * cost_per_guest

/-- Theorem stating the total cost of the wedding given the specified conditions. -/
theorem wedding_cost_theorem :
  wedding_cost 10000 500 50 60 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_wedding_cost_theorem_l4081_408125


namespace NUMINAMATH_CALUDE_circle_area_difference_l4081_408117

theorem circle_area_difference (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 21) (h₂ : r₂ = 31) :
  (r₃ ^ 2 = r₂ ^ 2 - r₁ ^ 2) → r₃ = 2 * Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l4081_408117


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l4081_408124

theorem kendall_driving_distance (mother_distance father_distance : ℝ) 
  (h1 : mother_distance = 0.17)
  (h2 : father_distance = 0.5) :
  mother_distance + father_distance = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l4081_408124


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l4081_408192

/-- The area of a semicircle that circumscribes a 2 × 3 rectangle with the longer side on the diameter -/
theorem semicircle_area_with_inscribed_rectangle : 
  ∀ (semicircle_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ),
    rectangle_width = 2 →
    rectangle_length = 3 →
    semicircle_area = (9 * Real.pi) / 4 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l4081_408192


namespace NUMINAMATH_CALUDE_exactly_one_first_class_product_l4081_408153

theorem exactly_one_first_class_product (p1 p2 : ℝ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/4) : 
  p1 * (1 - p2) + (1 - p1) * p2 = 5/12 := by
sorry

end NUMINAMATH_CALUDE_exactly_one_first_class_product_l4081_408153


namespace NUMINAMATH_CALUDE_root_in_interval_l4081_408182

-- Define the function f(x) = x^2 + 12x - 15
def f (x : ℝ) : ℝ := x^2 + 12*x - 15

-- State the theorem
theorem root_in_interval :
  (f 1.1 < 0) → (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l4081_408182


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l4081_408181

theorem triangle_max_perimeter : 
  ∀ x y z : ℕ,
  x = 4 * y →
  z = 20 →
  (x + y > z ∧ x + z > y ∧ y + z > x) →
  ∀ a b c : ℕ,
  a = 4 * b →
  c = 20 →
  (a + b > c ∧ a + c > b ∧ b + c > a) →
  x + y + z ≤ a + b + c →
  x + y + z ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l4081_408181


namespace NUMINAMATH_CALUDE_quadratic_with_complex_root_l4081_408145

theorem quadratic_with_complex_root (a b c : ℝ) :
  (∀ x : ℂ, a * x^2 + b * x + c = 0 ↔ x = -1 + 2*I ∨ x = -1 - 2*I) →
  a = 1 ∧ b = 2 ∧ c = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_with_complex_root_l4081_408145


namespace NUMINAMATH_CALUDE_remaining_money_l4081_408175

def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

theorem remaining_money : 
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l4081_408175


namespace NUMINAMATH_CALUDE_functions_properties_l4081_408139

/-- Given functions f and g with parameter a, prove monotonicity of g and range of b -/
theorem functions_properties (a : ℝ) (h : a < -1) :
  let f := fun (x : ℝ) ↦ x^3 / 3 - x^2 / 2 + a^2 / 2 - 1 / 3
  let g := fun (x : ℝ) ↦ a * Real.log (x + 1) - x^2 / 2 - a * x
  let g_deriv := fun (x : ℝ) ↦ a / (x + 1) - x - a
  let monotonic_intervals := (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (-a - 1), Set.Ioo 0 (-a - 1))
  let b := g (-a - 1) - f 1
  (∀ x ∈ monotonic_intervals.1, g_deriv x < 0) ∧
  (∀ x ∈ monotonic_intervals.2, g_deriv x > 0) ∧
  (∀ y : ℝ, y < 0 → ∃ x : ℝ, b = y) := by
  sorry


end NUMINAMATH_CALUDE_functions_properties_l4081_408139


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l4081_408195

theorem simple_interest_rate_percent : 
  ∀ (principal interest time rate : ℝ),
  principal = 800 →
  interest = 176 →
  time = 4 →
  interest = principal * rate * time / 100 →
  rate = 5.5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l4081_408195


namespace NUMINAMATH_CALUDE_product_of_roots_l4081_408147

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + 5 * a - 15 = 0) →
  (3 * b^3 - 9 * b^2 + 5 * b - 15 = 0) →
  (3 * c^3 - 9 * c^2 + 5 * c - 15 = 0) →
  a * b * c = 5 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l4081_408147


namespace NUMINAMATH_CALUDE_find_number_l4081_408104

theorem find_number : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  n = sum * quotient + 20 ∧ n / sum = quotient ∧ n % sum = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4081_408104


namespace NUMINAMATH_CALUDE_line_through_coefficients_l4081_408155

/-- Given two lines that intersect at (2,3), prove that the line passing through their coefficients has a specific equation -/
theorem line_through_coefficients 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : 2*a₁ + 3*b₁ + 1 = 0) 
  (h₂ : 2*a₂ + 3*b₂ + 1 = 0) :
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2*x + 3*y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_coefficients_l4081_408155


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4081_408171

theorem arithmetic_sequence_sum (a₁ d n : ℕ) (h : n > 0) : 
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  a₁ = 4 ∧ d = 5 ∧ n = 12 → S = 378 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4081_408171


namespace NUMINAMATH_CALUDE_min_value_x_l4081_408108

theorem min_value_x (x : ℝ) : 2 * (x + 1) ≥ x + 1 → x ≥ -1 ∧ ∀ y, (∀ z, 2 * (z + 1) ≥ z + 1 → z ≥ y) → y ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l4081_408108


namespace NUMINAMATH_CALUDE_intersection_condition_l4081_408134

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the line
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Define the intersection condition
def always_intersects (k : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola x y ∧ line k x y → x^2 - (k*x - 1)^2 = 4

-- State the theorem
theorem intersection_condition (k : ℝ) :
  always_intersects k ↔ (k = 1 ∨ k = -1 ∨ (-Real.sqrt 5 / 2 ≤ k ∧ k ≤ Real.sqrt 5 / 2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l4081_408134


namespace NUMINAMATH_CALUDE_library_books_count_library_books_count_proof_l4081_408118

theorem library_books_count : ℕ → Prop :=
  fun N =>
    let initial_issued := N / 17
    let transferred := 2000
    let new_issued := initial_issued + transferred
    (initial_issued = (N - initial_issued) / 16) ∧
    (new_issued = (N - new_issued) / 15) →
    N = 544000

-- The proof goes here
theorem library_books_count_proof : library_books_count 544000 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_library_books_count_proof_l4081_408118


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l4081_408166

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 42 →
  boys + girls = total_students →
  3 * girls = 4 * boys →
  girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l4081_408166


namespace NUMINAMATH_CALUDE_coyote_coins_proof_l4081_408184

/-- Represents the number of coins Coyote has after each crossing and payment -/
def coins_after_crossing (initial_coins : ℕ) (num_crossings : ℕ) : ℤ :=
  (3^num_crossings * initial_coins) - (50 * (3^num_crossings - 1) / 2)

/-- Theorem stating that Coyote ends up with 0 coins after 4 crossings if he starts with 25 coins -/
theorem coyote_coins_proof :
  coins_after_crossing 25 4 = 0 := by
  sorry

#eval coins_after_crossing 25 4

end NUMINAMATH_CALUDE_coyote_coins_proof_l4081_408184


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l4081_408140

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l4081_408140


namespace NUMINAMATH_CALUDE_fraction_power_product_l4081_408159

theorem fraction_power_product : (3/4)^4 * (1/5) = 81/1280 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l4081_408159


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l4081_408123

/-- Given a mixture of milk and water, proves that the initial ratio was 4:1 --/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) :
  total_volume = 45 →
  added_water = 3 →
  final_ratio = 3 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l4081_408123


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_exists_l4081_408167

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the midpoint
def is_midpoint (x1 y1 x2 y2 x0 y0 : ℝ) : Prop :=
  x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

theorem hyperbola_midpoint_exists :
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_hyperbola x1 y1 ∧
    is_on_hyperbola x2 y2 ∧
    is_midpoint x1 y1 x2 y2 (-1) (-4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_exists_l4081_408167


namespace NUMINAMATH_CALUDE_four_students_same_group_probability_l4081_408169

/-- The number of students in the school -/
def total_students : ℕ := 720

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- The probability of four specific students being assigned to the same lunch group -/
def prob_four_students_same_group : ℚ := prob_assigned_to_group ^ 3

theorem four_students_same_group_probability :
  prob_four_students_same_group = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_four_students_same_group_probability_l4081_408169


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l4081_408193

/-- An ellipse with given properties -/
structure Ellipse :=
  (A B E F : ℝ × ℝ)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4)
  (AF_length : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = 2 + Real.sqrt 3)

/-- A point on the ellipse satisfying the given condition -/
def PointOnEllipse (Γ : Ellipse) (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
  Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2) = 2

/-- The theorem to be proved -/
theorem ellipse_triangle_area (Γ : Ellipse) (P : ℝ × ℝ) (h : PointOnEllipse Γ P) :
  (1/2) * Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
         Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2) *
         Real.sin (Real.arccos (
           ((P.1 - Γ.E.1) * (P.1 - Γ.F.1) + (P.2 - Γ.E.2) * (P.2 - Γ.F.2)) /
           (Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
            Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2))
         )) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l4081_408193


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l4081_408154

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ - x₀*y₀ = 0 ∧ x₀ + 2*y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l4081_408154


namespace NUMINAMATH_CALUDE_ratio_sum_to_base_l4081_408116

theorem ratio_sum_to_base (a b : ℚ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_base_l4081_408116


namespace NUMINAMATH_CALUDE_cake_recipe_difference_l4081_408165

theorem cake_recipe_difference (total_flour total_sugar flour_added : ℕ) : 
  total_flour = 9 → 
  total_sugar = 11 → 
  flour_added = 4 → 
  total_sugar - (total_flour - flour_added) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cake_recipe_difference_l4081_408165


namespace NUMINAMATH_CALUDE_car_trip_distance_l4081_408144

theorem car_trip_distance (D : ℝ) : 
  (1/2 : ℝ) * D + (1/4 : ℝ) * ((1/2 : ℝ) * D) + 105 = D → D = 280 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l4081_408144


namespace NUMINAMATH_CALUDE_cat_food_cans_l4081_408196

/-- The number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := sorry

/-- The number of packages of cat food -/
def cat_packages : ℕ := 6

/-- The number of packages of dog food -/
def dog_packages : ℕ := 2

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 3

theorem cat_food_cans : 
  cat_packages * cans_per_cat_package = 
  dog_packages * cans_per_dog_package + 48 ∧ 
  cans_per_cat_package = 9 :=
sorry

end NUMINAMATH_CALUDE_cat_food_cans_l4081_408196


namespace NUMINAMATH_CALUDE_line_parabola_intersection_length_l4081_408185

/-- Given a line y = kx - k intersecting the parabola y^2 = 4x at points A and B,
    if the midpoint of AB is 3 units from the y-axis, then the length of AB is 8 units. -/
theorem line_parabola_intersection_length (k : ℝ) (A B : ℝ × ℝ) : 
  (∀ x y, y = k * x - k → y^2 = 4 * x → (x, y) = A ∨ (x, y) = B) →  -- Line intersects parabola at A and B
  ((A.1 + B.1) / 2 = 3) →                                           -- Midpoint is 3 units from y-axis
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=                  -- Length of AB is 8 units
by sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_length_l4081_408185


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l4081_408189

theorem average_age_when_youngest_born 
  (n : ℕ) 
  (current_avg : ℝ) 
  (youngest_age : ℝ) 
  (sum_others_at_birth : ℝ) 
  (h1 : n = 7) 
  (h2 : current_avg = 30) 
  (h3 : youngest_age = 6) 
  (h4 : sum_others_at_birth = 150) : 
  (sum_others_at_birth / n : ℝ) = 150 / 7 := by
sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l4081_408189


namespace NUMINAMATH_CALUDE_fourth_power_sum_l4081_408137

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59/3 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l4081_408137


namespace NUMINAMATH_CALUDE_not_perfect_square_zero_six_l4081_408131

/-- A number composed only of digits 0 and 6 -/
def DigitsZeroSix (m : ℕ) : Prop :=
  ∀ d, d ∈ m.digits 10 → d = 0 ∨ d = 6

/-- The sum of digits of a natural number -/
def DigitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem not_perfect_square_zero_six (m : ℕ) (h : DigitsZeroSix m) : 
  ¬∃ k : ℕ, m = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_zero_six_l4081_408131


namespace NUMINAMATH_CALUDE_red_cars_count_l4081_408122

/-- Represents the car rental problem --/
structure CarRental where
  num_white_cars : ℕ
  white_car_cost : ℕ
  red_car_cost : ℕ
  rental_duration : ℕ
  total_earnings : ℕ

/-- Calculates the number of red cars given the rental information --/
def calculate_red_cars (rental : CarRental) : ℕ :=
  (rental.total_earnings - rental.num_white_cars * rental.white_car_cost * rental.rental_duration) /
  (rental.red_car_cost * rental.rental_duration)

/-- Theorem stating that the number of red cars is 3 --/
theorem red_cars_count (rental : CarRental)
  (h1 : rental.num_white_cars = 2)
  (h2 : rental.white_car_cost = 2)
  (h3 : rental.red_car_cost = 3)
  (h4 : rental.rental_duration = 180)
  (h5 : rental.total_earnings = 2340) :
  calculate_red_cars rental = 3 := by
  sorry

#eval calculate_red_cars { num_white_cars := 2, white_car_cost := 2, red_car_cost := 3, rental_duration := 180, total_earnings := 2340 }

end NUMINAMATH_CALUDE_red_cars_count_l4081_408122


namespace NUMINAMATH_CALUDE_inequality_proof_l4081_408150

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

theorem inequality_proof (m : ℝ) (a b c : ℝ) 
  (h1 : Set.Icc 0 2 = {x | f m (x + 1) ≥ 0})
  (h2 : 1/a + 1/(2*b) + 1/(3*c) = m) : 
  a + 2*b + 3*c ≥ 9 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4081_408150


namespace NUMINAMATH_CALUDE_worker_completion_time_l4081_408113

/-- Given workers x and y, where x can complete a job in 40 days,
    x works for 8 days, and y finishes the remaining work in 16 days,
    prove that y can complete the entire job alone in 20 days. -/
theorem worker_completion_time
  (x_completion_time : ℕ)
  (x_work_days : ℕ)
  (y_completion_time_for_remainder : ℕ)
  (h1 : x_completion_time = 40)
  (h2 : x_work_days = 8)
  (h3 : y_completion_time_for_remainder = 16) :
  (y_completion_time_for_remainder * x_completion_time) / 
  (x_completion_time - x_work_days) = 20 :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l4081_408113


namespace NUMINAMATH_CALUDE_basketball_handshakes_l4081_408188

theorem basketball_handshakes : 
  let team_size : ℕ := 6
  let referee_count : ℕ := 3
  let inter_team_handshakes := team_size * team_size
  let player_referee_handshakes := (2 * team_size) * referee_count
  inter_team_handshakes + player_referee_handshakes = 72 :=
by sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l4081_408188


namespace NUMINAMATH_CALUDE_riley_time_outside_l4081_408164

theorem riley_time_outside (D : ℝ) (jonsey_awake : ℝ) (jonsey_outside : ℝ) (riley_awake : ℝ) (inside_time : ℝ) :
  D = 24 →
  jonsey_awake = (2/3) * D →
  jonsey_outside = (1/2) * jonsey_awake →
  riley_awake = (3/4) * D →
  jonsey_awake - jonsey_outside + riley_awake - (riley_awake * (8/9)) = inside_time →
  inside_time = 10 →
  riley_awake * (8/9) = riley_awake - (inside_time - (jonsey_awake - jonsey_outside)) :=
by sorry

end NUMINAMATH_CALUDE_riley_time_outside_l4081_408164


namespace NUMINAMATH_CALUDE_root_sum_fraction_equality_l4081_408176

theorem root_sum_fraction_equality (r s t : ℝ) : 
  r^3 - 6*r^2 + 11*r - 6 = 0 → 
  s^3 - 6*s^2 + 11*s - 6 = 0 → 
  t^3 - 6*t^2 + 11*t - 6 = 0 → 
  (r+s)/t + (s+t)/r + (t+r)/s = 25/3 :=
by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_equality_l4081_408176


namespace NUMINAMATH_CALUDE_remainder_problem_l4081_408128

theorem remainder_problem (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : n % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4081_408128
