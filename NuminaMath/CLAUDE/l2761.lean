import Mathlib

namespace exam_pass_count_l2761_276198

theorem exam_pass_count (total : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) :
  total = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ pass_count : ℕ,
    pass_count = 100 ∧
    pass_count ≤ total ∧
    (pass_count : ℚ) * avg_pass + (total - pass_count : ℚ) * avg_fail = (total : ℚ) * avg_all :=
by sorry

end exam_pass_count_l2761_276198


namespace factorial_ones_divisibility_l2761_276189

/-- Definition of [n]! -/
def factorial_ones (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (factorial_ones k) * (Nat.ofDigits 2 (List.replicate (k + 1) 1))

/-- Theorem stating that [n+m]! is divisible by [n]! · [m]! -/
theorem factorial_ones_divisibility (n m : ℕ) :
  ∃ k : ℕ, factorial_ones (n + m) = k * (factorial_ones n * factorial_ones m) := by
  sorry


end factorial_ones_divisibility_l2761_276189


namespace cross_out_all_stars_star_remains_uncrossed_l2761_276167

/-- Represents a 2n × 2n table with stars -/
structure StarTable (n : ℕ) where
  stars : Finset (Fin (2*n) × Fin (2*n))

/-- Theorem for part (a) -/
theorem cross_out_all_stars (n : ℕ) (table : StarTable n) 
  (h : table.stars.card = 3*n) :
  ∃ (rows columns : Finset (Fin (2*n))),
    rows.card = n ∧ 
    columns.card = n ∧
    (∀ star ∈ table.stars, star.1 ∈ rows ∨ star.2 ∈ columns) :=
sorry

/-- Theorem for part (b) -/
theorem star_remains_uncrossed (n : ℕ) (table : StarTable n)
  (h : table.stars.card = 3*n + 1) :
  ∀ (rows columns : Finset (Fin (2*n))),
    rows.card = n →
    columns.card = n →
    ∃ star ∈ table.stars, star.1 ∉ rows ∧ star.2 ∉ columns :=
sorry

end cross_out_all_stars_star_remains_uncrossed_l2761_276167


namespace race_distance_proof_l2761_276127

/-- The distance of the race where B beats C -/
def race_distance : ℝ := 800

theorem race_distance_proof :
  ∀ (v_a v_b v_c : ℝ),  -- speeds of A, B, and C
  v_a > 0 ∧ v_b > 0 ∧ v_c > 0 →  -- positive speeds
  (1000 / v_a = 900 / v_b) →  -- A beats B by 100m in 1000m race
  (race_distance / v_b = (race_distance - 100) / v_c) →  -- B beats C by 100m in race_distance
  (1000 / v_a = 787.5 / v_c) →  -- A beats C by 212.5m in 1000m race
  race_distance = 800 := by
sorry

end race_distance_proof_l2761_276127


namespace simplify_expression_l2761_276111

theorem simplify_expression (x : ℝ) : (3 * x + 25) + (200 * x - 50) = 203 * x - 25 := by
  sorry

end simplify_expression_l2761_276111


namespace shelbys_driving_time_l2761_276192

/-- Shelby's driving problem -/
theorem shelbys_driving_time (speed_sun speed_rain : ℝ) (total_time total_distance : ℝ) 
  (h1 : speed_sun = 30)
  (h2 : speed_rain = 20)
  (h3 : total_time = 40)
  (h4 : total_distance = 16)
  (h5 : speed_sun > 0 ∧ speed_rain > 0) :
  ∃ (time_rain : ℝ), 
    time_rain = 24 ∧ 
    time_rain > 0 ∧ 
    time_rain < total_time ∧
    (speed_sun * (total_time - time_rain) / 60 + speed_rain * time_rain / 60 = total_distance) :=
by sorry

end shelbys_driving_time_l2761_276192


namespace f_even_implies_a_zero_f_not_odd_l2761_276120

/-- Definition of the function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |x - a| + 1

/-- Theorem 1: If f is even, then a = 0 -/
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

/-- Theorem 2: f is not odd for any real a -/
theorem f_not_odd (a : ℝ) :
  ¬(∀ x : ℝ, f a (-x) = -(f a x)) := by sorry

end f_even_implies_a_zero_f_not_odd_l2761_276120


namespace sequence_ratio_l2761_276109

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (-2 : ℝ) - a₁ = a₁ - a₂ ∧ 
  a₂ - a₁ = a₁ - (-8 : ℝ) ∧
  (-2 : ℝ) * b₁ = b₁ * b₂ ∧
  b₁ * b₂ = b₂ * b₃ ∧
  b₂ * b₃ = b₃ * (-8 : ℝ) →
  (a₂ - a₁) / b₂ = 1/2 := by
sorry

end sequence_ratio_l2761_276109


namespace f_minus_one_equals_eight_l2761_276153

def f (x : ℝ) (c : ℝ) := x^2 + c

theorem f_minus_one_equals_eight (c : ℝ) (h : f 1 c = 8) : f (-1) c = 8 := by
  sorry

end f_minus_one_equals_eight_l2761_276153


namespace find_x_when_y_is_8_l2761_276180

-- Define the relationship between x and y
def varies_directly (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k * Real.sqrt x

-- State the theorem
theorem find_x_when_y_is_8 :
  ∀ x₀ y₀ x y : ℝ,
  varies_directly x₀ y₀ →
  varies_directly x y →
  x₀ = 3 →
  y₀ = 2 →
  y = 8 →
  x = 48 :=
by
  sorry

end find_x_when_y_is_8_l2761_276180


namespace vectors_not_collinear_l2761_276125

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are not collinear -/
theorem vectors_not_collinear (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, -2, 3))
  (hb : b = (3, 0, -1)) : 
  ¬ (∃ (k : ℝ), (2 • a + 4 • b) = k • (3 • b - a)) := by
  sorry

end vectors_not_collinear_l2761_276125


namespace inequality_system_solution_set_l2761_276141

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | 3 * x + 5 ≥ -1 ∧ 3 - x > (1/2) * x}
  S = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end inequality_system_solution_set_l2761_276141


namespace no_integer_solution_l2761_276166

theorem no_integer_solution : 
  ¬ ∃ (x y z : ℤ), (x - y)^3 + (y - z)^3 + (z - x)^3 = 2011 := by
  sorry

end no_integer_solution_l2761_276166


namespace ace_spade_probability_l2761_276129

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing an Ace as the first card and a spade as the second card -/
def prob_ace_then_spade : ℚ :=
  (NumAces / StandardDeck) * (NumSpades / (StandardDeck - 1))

theorem ace_spade_probability :
  prob_ace_then_spade = 3 / 127 := by
  sorry

end ace_spade_probability_l2761_276129


namespace sector_max_area_l2761_276149

theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 4) :
  (1/2) * l * r ≤ 1 := by
  sorry

end sector_max_area_l2761_276149


namespace dust_storm_coverage_l2761_276116

/-- The dust storm problem -/
theorem dust_storm_coverage (total_prairie : ℕ) (untouched : ℕ) (covered : ℕ) : 
  total_prairie = 64013 → untouched = 522 → covered = total_prairie - untouched → covered = 63491 := by
  sorry

end dust_storm_coverage_l2761_276116


namespace trajectory_of_Q_l2761_276128

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 1

-- Define the condition for point P
def P_on_C₂ (P : ℝ × ℝ) : Prop := C₂ P.1 P.2

-- Define the condition for point R
def R_on_C₁ (R : ℝ × ℝ) : Prop := C₁ R.1 R.2

-- Define the condition that R is on OP
def R_on_OP (O P R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ R.1 = t * P.1 ∧ R.2 = t * P.2

-- Define the condition for point Q
def Q_condition (O P Q R : ℝ × ℝ) : Prop :=
  (Q.1^2 + Q.2^2) * (P.1^2 + P.2^2) = (R.1^2 + R.2^2)^2

-- The main theorem
theorem trajectory_of_Q (O P Q R : ℝ × ℝ) :
  O = (0, 0) →
  P_on_C₂ P →
  R_on_C₁ R →
  R_on_OP O P R →
  Q_condition O P Q R →
  (Q.1 - 1/2)^2 + (Q.2 - 1/2)^2 = 1/2 :=
sorry

end trajectory_of_Q_l2761_276128


namespace jesse_blocks_theorem_l2761_276124

/-- The number of blocks Jesse used to build the building -/
def building_blocks : ℕ := 80

/-- The number of blocks Jesse used to build the farmhouse -/
def farmhouse_blocks : ℕ := 123

/-- The number of blocks Jesse used to build the fenced-in area -/
def fenced_area_blocks : ℕ := 57

/-- The number of blocks Jesse has left -/
def remaining_blocks : ℕ := 84

/-- The total number of blocks Jesse started with -/
def total_blocks : ℕ := building_blocks + farmhouse_blocks + fenced_area_blocks + remaining_blocks

theorem jesse_blocks_theorem : total_blocks = 344 := by
  sorry

end jesse_blocks_theorem_l2761_276124


namespace field_planted_fraction_l2761_276130

theorem field_planted_fraction (a b d : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) :
  let c := (a^2 + b^2).sqrt
  let x := (a * b * d) / (a^2 + b^2)
  let triangle_area := a * b / 2
  let square_area := x^2
  let planted_area := triangle_area - square_area
  a = 5 → b = 12 → d = 3 →
  planted_area / triangle_area = 52761 / 857430 := by
  sorry

end field_planted_fraction_l2761_276130


namespace power_of_power_l2761_276162

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2761_276162


namespace roots_eq1_roots_eq2_l2761_276146

-- Define the quadratic equations
def eq1 (x : ℝ) := x^2 - 2*x - 8
def eq2 (x : ℝ) := 2*x^2 - 4*x + 1

-- Theorem for the roots of the first equation
theorem roots_eq1 : 
  (eq1 4 = 0 ∧ eq1 (-2) = 0) ∧ 
  ∀ x : ℝ, eq1 x = 0 → x = 4 ∨ x = -2 := by sorry

-- Theorem for the roots of the second equation
theorem roots_eq2 : 
  (eq2 ((2 + Real.sqrt 2) / 2) = 0 ∧ eq2 ((2 - Real.sqrt 2) / 2) = 0) ∧ 
  ∀ x : ℝ, eq2 x = 0 → x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2 := by sorry

end roots_eq1_roots_eq2_l2761_276146


namespace E_parity_l2761_276178

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 1) + E n

def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem E_parity : (isEven (E 2021) ∧ ¬isEven (E 2022) ∧ ¬isEven (E 2023)) := by sorry

end E_parity_l2761_276178


namespace solve_for_y_l2761_276136

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end solve_for_y_l2761_276136


namespace triangle_relations_l2761_276161

/-- Given a triangle with area S, inradius r, exradii r_a, r_b, r_c, 
    side lengths a, b, c, circumradius R, and semiperimeter p -/
theorem triangle_relations (S r r_a r_b r_c a b c R : ℝ) 
  (h_positive : S > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_semiperimeter : ∃ p, p = (a + b + c) / 2) :
  (1 / r^3 - 1 / r_a^3 - 1 / r_b^3 - 1 / r_c^3 = 12 * R / S^2) ∧
  (a * (b + c) = (r + r_a) * (4 * R + r - r_a)) ∧
  (a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) := by
  sorry

end triangle_relations_l2761_276161


namespace division_problem_l2761_276183

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 760 → 
  divisor = 36 → 
  remainder = 4 → 
  dividend = divisor * quotient + remainder → 
  quotient = 21 := by
sorry

end division_problem_l2761_276183


namespace pascals_cycling_trip_l2761_276133

theorem pascals_cycling_trip (current_speed : ℝ) (speed_reduction : ℝ) (time_difference : ℝ) :
  current_speed = 8 →
  speed_reduction = 4 →
  time_difference = 16 →
  let reduced_speed := current_speed - speed_reduction
  let increased_speed := current_speed * 1.5
  ∃ (distance : ℝ), distance = 96 ∧
    distance / reduced_speed = distance / increased_speed + time_difference :=
by sorry

end pascals_cycling_trip_l2761_276133


namespace quadratic_root_factorization_l2761_276114

theorem quadratic_root_factorization 
  (a₀ a₁ a₂ x r s : ℝ) 
  (h₁ : a₂ ≠ 0) 
  (h₂ : a₀ ≠ 0) 
  (h₃ : a₀ + a₁ * r + a₂ * r^2 = 0) 
  (h₄ : a₀ + a₁ * s + a₂ * s^2 = 0) :
  a₀ + a₁ * x + a₂ * x^2 = a₀ * (1 - x / r) * (1 - x / s) :=
by sorry

end quadratic_root_factorization_l2761_276114


namespace additional_steps_day3_l2761_276159

def day1_steps : ℕ := 200 + 300

def day2_steps : ℕ := 2 * day1_steps

def total_steps : ℕ := 1600

theorem additional_steps_day3 : 
  total_steps - (day1_steps + day2_steps) = 100 := by sorry

end additional_steps_day3_l2761_276159


namespace refrigerator_transport_cost_l2761_276113

/-- Calculates the transport cost given the purchase details of a refrigerator --/
theorem refrigerator_transport_cost 
  (purchase_price_after_discount : ℕ)
  (discount_rate : ℚ)
  (installation_cost : ℕ)
  (selling_price_for_profit : ℕ) :
  purchase_price_after_discount = 12500 →
  discount_rate = 1/5 →
  installation_cost = 250 →
  selling_price_for_profit = 18560 →
  (purchase_price_after_discount / (1 - discount_rate) * (1 + 4/25) : ℚ) = selling_price_for_profit →
  (selling_price_for_profit : ℚ) - purchase_price_after_discount - installation_cost = 5810 :=
by sorry

end refrigerator_transport_cost_l2761_276113


namespace cos_20_minus_cos_40_l2761_276101

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end cos_20_minus_cos_40_l2761_276101


namespace evaluate_expression_l2761_276150

theorem evaluate_expression : 3 * Real.sqrt 32 + 2 * Real.sqrt 50 = 22 * Real.sqrt 2 := by
  sorry

end evaluate_expression_l2761_276150


namespace function_monotonicity_implies_a_value_l2761_276117

/-- A function f(x) = x^2 - ax that is decreasing on (-∞, 2] and increasing on (2, +∞) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - a*x

/-- The function f is decreasing on (-∞, 2] -/
def decreasing_on_left (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The function f is increasing on (2, +∞) -/
def increasing_on_right (a : ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f a x < f a y

/-- If f(x) = x^2 - ax is decreasing on (-∞, 2] and increasing on (2, +∞), then a = 4 -/
theorem function_monotonicity_implies_a_value (a : ℝ) :
  decreasing_on_left a → increasing_on_right a → a = 4 := by sorry

end function_monotonicity_implies_a_value_l2761_276117


namespace conspiracy_split_l2761_276131

theorem conspiracy_split (S : Finset (Finset Nat)) :
  S.card = 6 →
  (∀ s ∈ S, s.card = 3) →
  (∃ T : Finset Nat, T ⊆ Finset.range 6 ∧ T.card = 3 ∧
    ∀ s ∈ S, (s ⊆ T → False) ∧ (s ⊆ (Finset.range 6 \ T) → False)) :=
by sorry

end conspiracy_split_l2761_276131


namespace total_crayons_l2761_276174

def initial_crayons : ℕ := 7
def added_crayons : ℕ := 3

theorem total_crayons : 
  initial_crayons + added_crayons = 10 := by sorry

end total_crayons_l2761_276174


namespace f_max_on_interval_f_greater_than_3x_solution_set_l2761_276155

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x + 2) * abs (x - 2)

-- Theorem for the maximum value of f(x) on [-3, 1]
theorem f_max_on_interval :
  ∃ (M : ℝ), M = 4 ∧ ∀ x ∈ Set.Icc (-3) 1, f x ≤ M :=
sorry

-- Theorem for the solution set of f(x) > 3x
theorem f_greater_than_3x_solution_set :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ (-4 < x ∧ x < 1)} :=
sorry

end f_max_on_interval_f_greater_than_3x_solution_set_l2761_276155


namespace parallel_line_intersection_not_always_parallel_l2761_276168

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel and intersection operations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (m n : Line)
variable (h_distinct_planes : α ≠ β)
variable (h_distinct_lines : m ≠ n)

-- State the theorem
theorem parallel_line_intersection_not_always_parallel :
  ¬(∀ (α β : Plane) (m n : Line),
    α ≠ β → m ≠ n →
    parallel m α → intersect α β n → parallel_lines m n) :=
sorry

end parallel_line_intersection_not_always_parallel_l2761_276168


namespace sandwich_cost_l2761_276103

/-- Proves that the cost of each sandwich is $5 --/
theorem sandwich_cost (num_sandwiches : ℕ) (paid : ℕ) (change : ℕ) :
  num_sandwiches = 3 ∧ paid = 20 ∧ change = 5 →
  (paid - change) / num_sandwiches = 5 :=
by
  sorry

#check sandwich_cost

end sandwich_cost_l2761_276103


namespace rent_is_1000_l2761_276112

/-- Calculates the rent given salary, remaining amount, and the relationship between rent and other expenses. -/
def calculate_rent (salary : ℕ) (remaining : ℕ) : ℕ :=
  let total_expenses := salary - remaining
  total_expenses / 3

/-- Proves that the rent is $1000 given the conditions -/
theorem rent_is_1000 (salary : ℕ) (remaining : ℕ) 
  (h1 : salary = 5000)
  (h2 : remaining = 2000)
  (h3 : calculate_rent salary remaining = 1000) : 
  calculate_rent salary remaining = 1000 := by
  sorry

#eval calculate_rent 5000 2000

end rent_is_1000_l2761_276112


namespace dining_bill_calculation_l2761_276134

theorem dining_bill_calculation (people : ℕ) (tip_percentage : ℚ) (individual_share : ℚ) :
  people = 8 →
  tip_percentage = 1/10 →
  individual_share = 191125/10000 →
  ∃ (original_bill : ℚ), 
    (original_bill * (1 + tip_percentage)) / people = individual_share ∧
    original_bill = 139 :=
by sorry

end dining_bill_calculation_l2761_276134


namespace multiplicative_inverse_mod_million_l2761_276137

def C : ℕ := 123456
def D : ℕ := 166666
def M : ℕ := 48

theorem multiplicative_inverse_mod_million :
  (M * (C * D)) % 1000000 = 1 :=
by sorry

end multiplicative_inverse_mod_million_l2761_276137


namespace triangle_inequality_l2761_276144

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
by sorry

end triangle_inequality_l2761_276144


namespace consecutive_digits_sum_divisibility_l2761_276177

/-- Given four consecutive digits p, q, r, s, the sum of pqrs and srqp is divisible by 1111 -/
theorem consecutive_digits_sum_divisibility (p : ℕ) (h1 : p < 7) :
  ∃ (k : ℕ), 1000 * p + 100 * (p + 1) + 10 * (p + 2) + (p + 3) +
             1000 * (p + 3) + 100 * (p + 2) + 10 * (p + 1) + p = 1111 * k := by
  sorry

#check consecutive_digits_sum_divisibility

end consecutive_digits_sum_divisibility_l2761_276177


namespace second_divisor_problem_l2761_276173

theorem second_divisor_problem (N : ℕ) (D : ℕ) : 
  N % 35 = 25 → N % D = 4 → D = 31 := by
sorry

end second_divisor_problem_l2761_276173


namespace coin_division_problem_l2761_276182

theorem coin_division_problem : 
  ∃ (n : ℕ), n > 0 ∧ 
  n % 8 = 6 ∧ 
  n % 7 = 5 ∧ 
  n % 9 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 8 = 6 → m % 7 = 5 → m ≥ n) := by
  sorry

end coin_division_problem_l2761_276182


namespace spherical_triangle_area_l2761_276122

/-- The area of a spherical triangle formed by the intersection of a sphere with a trihedral angle -/
theorem spherical_triangle_area 
  (R : ℝ) 
  (α β γ : ℝ) 
  (h_positive : R > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_vertex_center : True)  -- Represents the condition that the vertex coincides with the sphere's center
  : ∃ (S_Δ : ℝ), S_Δ = R^2 * (α + β + γ - Real.pi) :=
sorry

end spherical_triangle_area_l2761_276122


namespace sum_equation_l2761_276172

theorem sum_equation : 27474 + 3699 + 1985 - 2047 = 31111 := by
  sorry

end sum_equation_l2761_276172


namespace ball_motion_time_formula_l2761_276197

/-- Represents the motion of a ball thrown upward -/
structure BallMotion where
  h : ℝ     -- Initial height
  V₀ : ℝ    -- Initial velocity
  g : ℝ     -- Gravitational acceleration
  t : ℝ     -- Time
  V : ℝ     -- Final velocity
  S : ℝ     -- Displacement

/-- The theorem stating the relationship between time, displacement, velocities, and height -/
theorem ball_motion_time_formula (b : BallMotion) 
  (hS : b.S = b.h + (1/2) * b.g * b.t^2 + b.V₀ * b.t)
  (hV : b.V = b.g * b.t + b.V₀) :
  b.t = (2 * (b.S - b.h)) / (b.V + b.V₀) :=
by sorry

end ball_motion_time_formula_l2761_276197


namespace a_2016_value_l2761_276181

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem a_2016_value : a 2016 = 3024 + Real.sqrt 3 := by
  sorry

end a_2016_value_l2761_276181


namespace triangle_equilateral_l2761_276115

theorem triangle_equilateral (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_condition : a^2 + b^2 + c^2 - a*b - b*c - a*c = 0) : 
  a = b ∧ b = c := by
  sorry

end triangle_equilateral_l2761_276115


namespace six_player_tournament_games_l2761_276105

/-- The number of games in a chess tournament where each player plays twice with every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 6 players, where each player plays twice with every other player, the total number of games played is 60. -/
theorem six_player_tournament_games :
  tournament_games 6 = 30 ∧ 2 * tournament_games 6 = 60 := by
  sorry

end six_player_tournament_games_l2761_276105


namespace zenobius_had_more_descendants_l2761_276151

/-- Calculates the total number of descendants for King Pafnutius -/
def pafnutius_descendants : ℕ :=
  2 + 60 * 2 + 20 * 1

/-- Calculates the total number of descendants for King Zenobius -/
def zenobius_descendants : ℕ :=
  4 + 35 * 3 + 35 * 1

/-- Proves that King Zenobius had more descendants than King Pafnutius -/
theorem zenobius_had_more_descendants :
  zenobius_descendants > pafnutius_descendants :=
by sorry

end zenobius_had_more_descendants_l2761_276151


namespace train_length_train_length_proof_l2761_276118

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length 
  (train_speed : Real) 
  (bridge_crossing_time : Real) 
  (bridge_length : Real) : Real :=
  let total_distance := train_speed * (1000 / 3600) * bridge_crossing_time
  total_distance - bridge_length

/-- Proves that a train traveling at 45 km/hr that crosses a 250 m bridge in 30 seconds has a length of 125 m. -/
theorem train_length_proof :
  train_length 45 30 250 = 125 := by
  sorry

end train_length_train_length_proof_l2761_276118


namespace regular_decagon_diagonal_intersections_eq_choose_l2761_276185

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def regular_decagon_diagonal_intersections : ℕ := 210

/-- A regular decagon has 10 sides -/
def regular_decagon_sides : ℕ := 10

/-- Theorem: The number of distinct interior intersection points of diagonals 
    in a regular decagon is equal to the number of ways to choose 4 vertices from 10 -/
theorem regular_decagon_diagonal_intersections_eq_choose :
  regular_decagon_diagonal_intersections = Nat.choose regular_decagon_sides 4 := by
  sorry

#eval regular_decagon_diagonal_intersections
#eval Nat.choose regular_decagon_sides 4

end regular_decagon_diagonal_intersections_eq_choose_l2761_276185


namespace square_roots_problem_l2761_276195

theorem square_roots_problem (x : ℝ) (h : x > x - 6) :
  (x ^ 2 = (x - 6) ^ 2) → x ^ 2 = 9 := by
sorry

end square_roots_problem_l2761_276195


namespace complex_calculation_l2761_276157

theorem complex_calculation (A M N : ℂ) (Q : ℝ) :
  A = 5 - 2*I →
  M = -3 + 2*I →
  N = 3*I →
  Q = 3 →
  (A - M + N - Q) * I = 1 + 5*I :=
by sorry

end complex_calculation_l2761_276157


namespace y_derivative_l2761_276160

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt (49 * x^2 + 1) * Real.arctan (7 * x) - Real.log (7 * x + Real.sqrt (49 * x^2 + 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = (7 * Real.arctan (7 * x)) / (2 * Real.sqrt (49 * x^2 + 1)) :=
by sorry

end y_derivative_l2761_276160


namespace plan_y_more_cost_effective_l2761_276186

/-- The cost in cents for Plan X given m megabytes -/
def cost_x (m : ℕ) : ℕ := 5 * m

/-- The cost in cents for Plan Y given m megabytes -/
def cost_y (m : ℕ) : ℕ := 3000 + 3 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective -/
def min_megabytes : ℕ := 1501

theorem plan_y_more_cost_effective :
  ∀ m : ℕ, m ≥ min_megabytes → cost_y m < cost_x m ∧
  ∀ n : ℕ, n < min_megabytes → cost_y n ≥ cost_x n :=
by sorry

end plan_y_more_cost_effective_l2761_276186


namespace circle_area_with_chord_l2761_276193

theorem circle_area_with_chord (chord_length : ℝ) (center_to_chord : ℝ) (area : ℝ) : 
  chord_length = 10 →
  center_to_chord = 5 →
  area = π * (center_to_chord^2 + (chord_length / 2)^2) →
  area = 50 * π := by
sorry

end circle_area_with_chord_l2761_276193


namespace find_number_l2761_276194

theorem find_number (N : ℚ) : (4 / 5 * N) + 18 = N / (4 / 5) → N = 40 := by
  sorry

end find_number_l2761_276194


namespace coin_arrangement_count_l2761_276126

/-- Represents the number of ways to arrange 5 gold and 5 silver coins -/
def colorArrangements : ℕ := Nat.choose 10 5

/-- Represents the number of valid face orientations for 10 coins -/
def validOrientations : ℕ := 144

/-- The total number of distinguishable arrangements -/
def totalArrangements : ℕ := colorArrangements * validOrientations

/-- Theorem stating the number of distinguishable arrangements -/
theorem coin_arrangement_count :
  totalArrangements = 36288 :=
sorry

end coin_arrangement_count_l2761_276126


namespace stairs_climbed_l2761_276184

theorem stairs_climbed (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 4872 → 
  julia_stairs = Int.floor (2 * Real.sqrt (jonny_stairs / 2) + 15) → 
  jonny_stairs + julia_stairs = 4986 := by
sorry

end stairs_climbed_l2761_276184


namespace planter_pots_cost_l2761_276107

/-- Calculates the total cost of filling planter pots with plants, including sales tax. -/
def total_cost (num_pots : ℕ) (palm_fern_cost creeping_jenny_cost geranium_cost elephant_ear_cost purple_grass_cost : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let plant_cost_per_pot := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost + 2 * elephant_ear_cost + 3 * purple_grass_cost
  let total_plant_cost := num_pots * plant_cost_per_pot
  let sales_tax := sales_tax_rate * total_plant_cost
  total_plant_cost + sales_tax

/-- Theorem stating that the total cost to fill 6 planter pots with the given plants and 7% sales tax is $494.34. -/
theorem planter_pots_cost : total_cost 6 15 4 3.5 7 6 (7/100) = 494.34 := by
  sorry

end planter_pots_cost_l2761_276107


namespace problem_statement_l2761_276156

theorem problem_statement (x y : ℝ) (θ : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_θ_range : π/4 < θ ∧ θ < π/2)
  (h_eq1 : Real.cos θ / x = Real.sin θ / y)
  (h_eq2 : Real.sin θ^2 / x^2 + Real.cos θ^2 / y^2 = 10 / (3 * (x^2 + y^2))) :
  (x + y)^2 / (x^2 + y^2) = (2 + Real.sqrt 3) / 2 := by
  sorry

end problem_statement_l2761_276156


namespace triangle_inequality_l2761_276188

theorem triangle_inequality (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_perimeter : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
  sorry

end triangle_inequality_l2761_276188


namespace polar_point_equivalence_l2761_276147

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to standard form where r > 0 and 0 ≤ θ < 2π -/
def toStandardForm (p : PolarPoint) : PolarPoint :=
  sorry

theorem polar_point_equivalence :
  let p := PolarPoint.mk (-4) (5 * Real.pi / 6)
  let standardP := toStandardForm p
  standardP.r = 4 ∧ standardP.θ = 11 * Real.pi / 6 := by
  sorry

end polar_point_equivalence_l2761_276147


namespace ones_digit_73_power_l2761_276119

theorem ones_digit_73_power (n : ℕ) : 
  (73^n % 10 = 7) ↔ (n % 4 = 3) := by
sorry

end ones_digit_73_power_l2761_276119


namespace towel_bleaching_l2761_276138

theorem towel_bleaching (original_length original_breadth : ℝ) 
  (h_positive : original_length > 0 ∧ original_breadth > 0) :
  let new_length := 0.7 * original_length
  let new_area := 0.42 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.6 * original_breadth ∧
    new_length * new_breadth = new_area :=
by sorry

end towel_bleaching_l2761_276138


namespace three_x_squared_y_squared_l2761_276171

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
  sorry

end three_x_squared_y_squared_l2761_276171


namespace lcm_of_20_45_75_l2761_276142

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by
  sorry

end lcm_of_20_45_75_l2761_276142


namespace smallest_prime_with_digit_sum_23_l2761_276199

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 23 → p ≤ q :=
by sorry

end smallest_prime_with_digit_sum_23_l2761_276199


namespace two_a_minus_b_value_l2761_276158

theorem two_a_minus_b_value (a b : ℝ) 
  (ha : |a| = 4)
  (hb : |b| = 5)
  (hab : |a + b| = -(a + b)) :
  2*a - b = 13 ∨ 2*a - b = -3 := by
sorry

end two_a_minus_b_value_l2761_276158


namespace natural_number_equation_solutions_l2761_276108

theorem natural_number_equation_solutions :
  ∀ a b : ℕ,
  a^b + b^a = 10 * b^(a-2) + 100 ↔ (a = 109 ∧ b = 1) ∨ (a = 7 ∧ b = 3) :=
by sorry

end natural_number_equation_solutions_l2761_276108


namespace expression_proof_l2761_276175

theorem expression_proof (a b E : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : E / (3 * a - 2 * b) = 3) : 
  E = 6 * b := by
sorry

end expression_proof_l2761_276175


namespace arithmetic_sequence_sum_l2761_276187

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of specific terms in the sequence -/
def SpecificSum (a : ℕ → ℝ) : ℝ :=
  a 2 + a 4 + a 9 + a 11

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SpecificSum a = 32 → a 6 + a 7 = 16 := by
  sorry

end arithmetic_sequence_sum_l2761_276187


namespace initial_ace_cards_l2761_276148

/-- Represents the number of cards Nell has --/
structure NellCards where
  initialBaseballCards : ℕ
  finalBaseballCards : ℕ
  finalAceCards : ℕ
  aceBaseballDifference : ℕ

/-- Theorem stating the initial number of Ace cards Nell had --/
theorem initial_ace_cards (n : NellCards) 
  (h1 : n.initialBaseballCards = 239)
  (h2 : n.finalBaseballCards = 111)
  (h3 : n.finalAceCards = 376)
  (h4 : n.aceBaseballDifference = 265)
  (h5 : n.finalAceCards - n.finalBaseballCards = n.aceBaseballDifference) :
  n.finalAceCards + (n.initialBaseballCards - n.finalBaseballCards) = 504 := by
  sorry

end initial_ace_cards_l2761_276148


namespace fraction_absolute_value_less_than_one_l2761_276165

theorem fraction_absolute_value_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |((x - y) / (1 - x * y))| < 1 := by
sorry

end fraction_absolute_value_less_than_one_l2761_276165


namespace semicircle_radius_is_ten_l2761_276190

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The height of the triangle, which is equal to the length of its legs -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base of the triangle is 20 units -/
  base_eq : base = 20
  /-- The semicircle's diameter is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base

/-- The radius of the inscribed semicircle is 10 units -/
theorem semicircle_radius_is_ten (t : IsoscelesTriangleWithSemicircle) : t.radius = 10 := by
  sorry

end semicircle_radius_is_ten_l2761_276190


namespace cuboid_height_proof_l2761_276196

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The height of a cuboid with surface area 700 m², length 12 m, and width 14 m is 7 m -/
theorem cuboid_height_proof (surfaceArea length width : ℝ) 
  (hsa : surfaceArea = 700)
  (hl : length = 12)
  (hw : width = 14) :
  ∃ height : ℝ, cuboidSurfaceArea length width height = surfaceArea ∧ height = 7 := by
  sorry

end cuboid_height_proof_l2761_276196


namespace tangent_line_correct_l2761_276170

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The point through which the tangent line passes -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the proposed tangent line -/
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_correct :
  (f point.1 = point.2) ∧ 
  (∀ x y : ℝ, tangent_line x y → y - point.2 = f' point.1 * (x - point.1)) ∧
  (∀ x : ℝ, x ≠ point.1 → f x ≠ (f' point.1) * (x - point.1) + point.2) :=
sorry

end tangent_line_correct_l2761_276170


namespace parabola_constant_l2761_276143

/-- A parabola with equation x = dy² + ey + f, vertex at (3, -1), and passing through (5, 1) has f = 7/2 -/
theorem parabola_constant (d e f : ℝ) : 
  (∀ y : ℝ, 3 = d * (-1)^2 + e * (-1) + f) →  -- vertex condition
  (5 = d * 1^2 + e * 1 + f) →                 -- point condition
  (∀ y : ℝ, 3 = d * (y + 1)^2 + 3) →          -- vertex form
  f = 7/2 := by sorry

end parabola_constant_l2761_276143


namespace marcel_total_cost_l2761_276140

/-- The cost of Marcel's purchases -/
def total_cost (pen_price briefcase_price : ℝ) : ℝ :=
  pen_price + briefcase_price

/-- Theorem: Marcel's total cost for a pen and briefcase is $24 -/
theorem marcel_total_cost :
  ∃ (pen_price briefcase_price : ℝ),
    pen_price = 4 ∧
    briefcase_price = 5 * pen_price ∧
    total_cost pen_price briefcase_price = 24 := by
  sorry

end marcel_total_cost_l2761_276140


namespace second_cube_surface_area_l2761_276100

theorem second_cube_surface_area (v1 v2 : ℝ) (h1 : v1 = 16) (h2 : v2 = 4 * v1) :
  6 * (v2 ^ (1/3 : ℝ))^2 = 96 := by
  sorry

end second_cube_surface_area_l2761_276100


namespace square_area_14m_l2761_276176

/-- The area of a square with side length 14 meters is 196 square meters. -/
theorem square_area_14m (side_length : ℝ) (h : side_length = 14) : 
  side_length * side_length = 196 := by
  sorry

end square_area_14m_l2761_276176


namespace parabola_line_intersection_l2761_276179

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Check if two points are perpendicular with respect to the origin -/
def perpendicular (a b : Point) : Prop :=
  a.x * b.x + a.y * b.y = 0

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  point.y = line.m * point.x + line.b

/-- The main theorem -/
theorem parabola_line_intersection 
  (C : Parabola) 
  (F : Point)
  (l : Line)
  (A B : Point)
  (h1 : F.x = 1/2 ∧ F.y = 0)
  (h2 : l.m = 2)
  (h3 : onParabola A C ∧ onParabola B C)
  (h4 : onLine A l ∧ onLine B l)
  (h5 : A ≠ ⟨0, 0⟩ ∧ B ≠ ⟨0, 0⟩)
  (h6 : perpendicular A B) :
  C.p = 1 ∧ l.b = -4 :=
sorry

end parabola_line_intersection_l2761_276179


namespace strongest_correlation_l2761_276152

-- Define the type for a pair of observations
structure Observation where
  n : ℕ
  r : ℝ

-- Define the four given observations
def obs1 : Observation := ⟨10, 0.9533⟩
def obs2 : Observation := ⟨15, 0.3012⟩
def obs3 : Observation := ⟨17, 0.9991⟩
def obs4 : Observation := ⟨3, 0.9950⟩

-- Define a function to check if an observation indicates strong linear correlation
def isStrongCorrelation (obs : Observation) : Prop :=
  abs obs.r > 0.95

-- Theorem stating that obs1 and obs3 have the strongest linear correlation
theorem strongest_correlation :
  isStrongCorrelation obs1 ∧ isStrongCorrelation obs3 ∧
  ¬isStrongCorrelation obs2 ∧ ¬isStrongCorrelation obs4 :=
sorry

end strongest_correlation_l2761_276152


namespace nancy_pears_l2761_276139

/-- Given that Alyssa picked 42 pears and the total number of pears picked was 59,
    prove that Nancy picked 17 pears. -/
theorem nancy_pears (alyssa_pears total_pears : ℕ) 
  (h1 : alyssa_pears = 42)
  (h2 : total_pears = 59) :
  total_pears - alyssa_pears = 17 := by
  sorry

end nancy_pears_l2761_276139


namespace probability_divisible_by_15_l2761_276123

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 9}

/-- The number of digits -/
def n : Nat := 6

/-- The condition for divisibility by 15 -/
def divisible_by_15 (num : Nat) : Prop := num % 15 = 0

/-- The set of all possible six-digit numbers formed by the given digits -/
def all_numbers : Finset Nat := sorry

/-- The set of all six-digit numbers formed by the given digits that are divisible by 15 -/
def divisible_numbers : Finset Nat := sorry

/-- The probability of a randomly selected six-digit number being divisible by 15 -/
theorem probability_divisible_by_15 : 
  (Finset.card divisible_numbers : ℚ) / (Finset.card all_numbers : ℚ) = 1 / 6 := by sorry

end probability_divisible_by_15_l2761_276123


namespace rational_inequality_solution_l2761_276169

theorem rational_inequality_solution (x : ℝ) :
  (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6)) > 0 ↔ 
  x < 2 ∨ (4 < x ∧ x < 5) ∨ 6 < x :=
by sorry

end rational_inequality_solution_l2761_276169


namespace popsicle_stick_cost_l2761_276164

/-- Represents the cost of popsicle supplies in dollars -/
structure PopsicleSupplies where
  total_budget : ℚ
  mold_cost : ℚ
  juice_cost_per_bottle : ℚ
  popsicles_per_bottle : ℕ
  total_sticks : ℕ
  remaining_sticks : ℕ

/-- Calculates the cost of the pack of popsicle sticks -/
def stick_pack_cost (supplies : PopsicleSupplies) : ℚ :=
  supplies.total_budget - supplies.mold_cost - 
  (supplies.juice_cost_per_bottle * ((supplies.total_sticks - supplies.remaining_sticks) / supplies.popsicles_per_bottle))

/-- Theorem stating that the cost of the pack of popsicle sticks is $1 -/
theorem popsicle_stick_cost (supplies : PopsicleSupplies) 
  (h1 : supplies.total_budget = 10)
  (h2 : supplies.mold_cost = 3)
  (h3 : supplies.juice_cost_per_bottle = 2)
  (h4 : supplies.popsicles_per_bottle = 20)
  (h5 : supplies.total_sticks = 100)
  (h6 : supplies.remaining_sticks = 40) :
  stick_pack_cost supplies = 1 := by
  sorry

#eval stick_pack_cost { 
  total_budget := 10, 
  mold_cost := 3, 
  juice_cost_per_bottle := 2, 
  popsicles_per_bottle := 20, 
  total_sticks := 100, 
  remaining_sticks := 40 
}

end popsicle_stick_cost_l2761_276164


namespace quadratic_equation_solution_l2761_276191

theorem quadratic_equation_solution :
  let f (x : ℝ) := x^2 - 5*x + 1
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧
               x₂ = (5 - Real.sqrt 21) / 2 ∧
               f x₁ = 0 ∧ f x₂ = 0 ∧
               ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end quadratic_equation_solution_l2761_276191


namespace min_pool_cost_l2761_276154

/-- Represents the cost of constructing a rectangular pool -/
def pool_cost (l w h : ℝ) : ℝ :=
  120 * l * w + 80 * 2 * h * (l + w)

/-- The minimum cost of constructing a pool with given specifications -/
theorem min_pool_cost :
  ∀ l w : ℝ,
  l > 0 ∧ w > 0 →
  l * w * 2 = 8 →
  pool_cost l w 2 ≥ 1760 :=
sorry

end min_pool_cost_l2761_276154


namespace festival_attendance_l2761_276145

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900)
  (h3 : ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧
    (3 * girls) / 4 + (boys / 3) = festival_attendees) :
  ∃ (girls : ℕ), (3 * girls) / 4 = 720 := by
sorry

end festival_attendance_l2761_276145


namespace larger_triangle_equilateral_iff_l2761_276121

/-- Two identical right-angled triangles with angles α and β form a larger triangle when placed together with identical legs adjacent. -/
structure TrianglePair where
  α : Real
  β : Real
  right_angled : α + β = 90
  non_negative : 0 ≤ α ∧ 0 ≤ β

/-- The larger triangle formed by combining two identical right-angled triangles. -/
structure LargerTriangle where
  pair : TrianglePair
  side_a : Real
  side_b : Real
  side_c : Real
  angle_A : Real
  angle_B : Real
  angle_C : Real

/-- The larger triangle is equilateral if and only if the original right-angled triangles have α = 60° and β = 30°. -/
theorem larger_triangle_equilateral_iff (t : LargerTriangle) :
  (t.side_a = t.side_b ∧ t.side_b = t.side_c) ↔ (t.pair.α = 60 ∧ t.pair.β = 30) :=
sorry

end larger_triangle_equilateral_iff_l2761_276121


namespace quadratic_factorization_l2761_276163

theorem quadratic_factorization (a b : ℤ) : 
  (∀ x : ℝ, 12 * x^2 - 38 * x - 40 = (4 * x + a) * (3 * x + b)) → 
  a - 3 * b = 22 := by
  sorry

end quadratic_factorization_l2761_276163


namespace cube_volume_l2761_276104

theorem cube_volume (cube_diagonal : ℝ) (h : cube_diagonal = 6 * Real.sqrt 2) :
  ∃ (volume : ℝ), volume = 216 ∧ volume = (cube_diagonal / Real.sqrt 2) ^ 3 := by
  sorry

end cube_volume_l2761_276104


namespace inequality_proof_l2761_276135

theorem inequality_proof (x y : ℝ) : 
  ((x * y - y^2) / (x^2 + 4 * x + 5))^3 ≤ ((x^2 - x * y) / (x^2 + 4 * x + 5))^3 := by
  sorry

end inequality_proof_l2761_276135


namespace smallest_n_congruence_l2761_276132

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(528 * m ≡ 1068 * m [MOD 30])) ∧ 
  (528 * n ≡ 1068 * n [MOD 30]) :=
by sorry

end smallest_n_congruence_l2761_276132


namespace seed_germination_problem_l2761_276110

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 1/5 →
  total_germination_rate = 13/50 →
  (seeds_plot1 * germination_rate_plot1 + seeds_plot2 * (germination_rate_plot2 : ℚ)) / (seeds_plot1 + seeds_plot2) = total_germination_rate →
  (germination_rate_plot2 : ℚ) = 7/20 :=
by
  sorry

#check seed_germination_problem

end seed_germination_problem_l2761_276110


namespace function_property_l2761_276102

-- Define the function f
variable (f : ℝ → ℝ)
-- Define the point a
variable (a : ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x < y → x < a → y < a → f x < f y)
variable (h2 : ∀ x, f (x + a) = f (a - x))
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ < a ∧ a < x₂)
variable (h4 : |x₁ - a| < |x₂ - a|)

-- State the theorem
theorem function_property : f (2*a - x₁) > f (2*a - x₂) := by sorry

end function_property_l2761_276102


namespace unique_coin_distribution_l2761_276106

/-- A structure representing the coin distribution in the piggy bank -/
structure CoinDistribution where
  one_ruble : ℕ
  two_rubles : ℕ
  five_rubles : ℕ

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Theorem stating the unique solution to the coin distribution problem -/
theorem unique_coin_distribution : 
  ∃! (d : CoinDistribution), 
    d.one_ruble + d.two_rubles + d.five_rubles = 1000 ∧ 
    d.one_ruble + 2 * d.two_rubles + 5 * d.five_rubles = 2000 ∧
    is_prime d.one_ruble ∧
    d.one_ruble = 3 ∧ d.two_rubles = 996 ∧ d.five_rubles = 1 := by
  sorry


end unique_coin_distribution_l2761_276106
