import Mathlib

namespace class_7_highest_prob_l1737_173728

/-- The number of classes -/
def num_classes : ℕ := 12

/-- The probability of getting a sum of n when throwing two dice -/
def prob_sum (n : ℕ) : ℚ :=
  match n with
  | 2 => 1 / 36
  | 3 => 1 / 18
  | 4 => 1 / 12
  | 5 => 1 / 9
  | 6 => 5 / 36
  | 7 => 1 / 6
  | 8 => 5 / 36
  | 9 => 1 / 9
  | 10 => 1 / 12
  | 11 => 1 / 18
  | 12 => 1 / 36
  | _ => 0

/-- Theorem: Class 7 has the highest probability of being selected -/
theorem class_7_highest_prob :
  ∀ n : ℕ, 2 ≤ n → n ≤ num_classes → prob_sum n ≤ prob_sum 7 :=
by sorry

end class_7_highest_prob_l1737_173728


namespace f_decreasing_on_interval_l1737_173753

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_interval_l1737_173753


namespace probability_at_least_one_l1737_173714

theorem probability_at_least_one (A B : ℝ) (hA : A = 0.6) (hB : B = 0.7) 
  (h_independent : True) : 1 - (1 - A) * (1 - B) = 0.88 := by
  sorry

end probability_at_least_one_l1737_173714


namespace sufficient_not_necessary_condition_l1737_173776

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x, x^2 + 2*x > 0 ∧ ¬(x > 1)) :=
by sorry

end sufficient_not_necessary_condition_l1737_173776


namespace square_diff_cubed_l1737_173777

theorem square_diff_cubed : (5^2 - 4^2)^3 = 729 := by
  sorry

end square_diff_cubed_l1737_173777


namespace solve_pickle_problem_l1737_173786

def pickle_problem (total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar : ℕ) : Prop :=
  let used_vinegar := initial_vinegar - remaining_vinegar
  let filled_jars := used_vinegar / vinegar_per_jar
  let total_pickles := filled_jars * pickles_per_jar
  let pickles_per_cucumber := total_pickles / total_cucumbers
  pickles_per_cucumber = 4 ∧
  total_jars = 4 ∧
  total_cucumbers = 10 ∧
  initial_vinegar = 100 ∧
  pickles_per_jar = 12 ∧
  vinegar_per_jar = 10 ∧
  remaining_vinegar = 60

theorem solve_pickle_problem :
  ∃ (total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar : ℕ),
  pickle_problem total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar :=
by
  sorry

end solve_pickle_problem_l1737_173786


namespace car_cost_sharing_l1737_173760

theorem car_cost_sharing (total_cost : ℕ) (initial_friends : ℕ) (car_wash_earnings : ℕ) (final_friends : ℕ) : 
  total_cost = 1700 →
  initial_friends = 6 →
  car_wash_earnings = 500 →
  final_friends = 5 →
  (total_cost - car_wash_earnings) / final_friends - (total_cost - car_wash_earnings) / initial_friends = 40 := by
  sorry

end car_cost_sharing_l1737_173760


namespace nonzero_digits_count_l1737_173791

-- Define the fraction
def f : ℚ := 84 / (2^5 * 5^9)

-- Define a function to count non-zero digits after the decimal point
noncomputable def count_nonzero_digits_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem nonzero_digits_count :
  count_nonzero_digits_after_decimal f = 2 := by sorry

end nonzero_digits_count_l1737_173791


namespace exists_counterfeit_finding_algorithm_l1737_173796

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : Coin

/-- Represents the result of a weighing operation -/
inductive WeighResult
| balanced : WeighResult
| leftLighter : WeighResult
| rightLighter : WeighResult

/-- A function that simulates weighing two sets of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighResult :=
  sorry

/-- The type of an algorithm to find the counterfeit coin -/
def FindCounterfeitAlgorithm := List Coin → Coin

/-- Theorem stating that there exists an algorithm to find the counterfeit coin -/
theorem exists_counterfeit_finding_algorithm :
  ∃ (algo : FindCounterfeitAlgorithm),
    ∀ (coins : List Coin),
      coins.length = 9 →
      (∃! (c : Coin), c ∈ coins ∧ c = Coin.counterfeit) →
      algo coins = Coin.counterfeit :=
sorry

end exists_counterfeit_finding_algorithm_l1737_173796


namespace complex_fraction_power_eight_l1737_173704

theorem complex_fraction_power_eight :
  ((2 + 2 * Complex.I) / (2 - 2 * Complex.I)) ^ 8 = 1 := by
  sorry

end complex_fraction_power_eight_l1737_173704


namespace unique_solution_l1737_173707

/-- Represents a digit in the equation --/
def Digit := Fin 10

/-- The equation is valid if it satisfies all conditions --/
def is_valid_equation (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  100 * A.val + 10 * C.val + A.val + 
  100 * B.val + 10 * B.val + D.val = 
  1000 * A.val + 100 * B.val + 10 * C.val + D.val

/-- There exists a unique solution to the equation --/
theorem unique_solution : 
  ∃! (A B C D : Digit), is_valid_equation A B C D ∧ 
    A.val = 9 ∧ B.val = 8 ∧ C.val = 0 ∧ D.val = 1 :=
sorry

end unique_solution_l1737_173707


namespace scientific_notation_of_small_number_l1737_173703

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -5 :=
by sorry

end scientific_notation_of_small_number_l1737_173703


namespace inequality_proof_l1737_173780

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  a^3 + b^3 ≥ a^2*b + a*b^2 ∧ (1/a - 1)*(1/b - 1)*(1/c - 1) ≥ 8 := by
  sorry

end inequality_proof_l1737_173780


namespace systematic_sample_theorem_l1737_173725

/-- Systematic sampling function that returns true if the number is in the sample -/
def in_systematic_sample (total : ℕ) (sample_size : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (total / sample_size) + 1

/-- Theorem stating that in a systematic sample of size 5 from 60 numbered parts,
    if 4, 16, 40, and 52 are in the sample, then 28 must also be in the sample -/
theorem systematic_sample_theorem :
  let total := 60
  let sample_size := 5
  (in_systematic_sample total sample_size 4) →
  (in_systematic_sample total sample_size 16) →
  (in_systematic_sample total sample_size 40) →
  (in_systematic_sample total sample_size 52) →
  (in_systematic_sample total sample_size 28) :=
by
  sorry

#check systematic_sample_theorem

end systematic_sample_theorem_l1737_173725


namespace average_scissors_after_changes_l1737_173709

/-- Represents a drawer with scissors and pencils -/
structure Drawer where
  scissors : ℕ
  pencils : ℕ

/-- Calculates the average number of scissors in the drawers -/
def averageScissors (drawers : List Drawer) : ℚ :=
  (drawers.map (·.scissors)).sum / drawers.length

theorem average_scissors_after_changes : 
  let initialDrawers : List Drawer := [
    { scissors := 39, pencils := 22 },
    { scissors := 27, pencils := 54 },
    { scissors := 45, pencils := 33 }
  ]
  let scissorsAdded : List ℕ := [13, 7, 10]
  let finalDrawers := List.zipWith 
    (fun d a => { scissors := d.scissors + a, pencils := d.pencils }) 
    initialDrawers 
    scissorsAdded
  averageScissors finalDrawers = 47 := by
  sorry

end average_scissors_after_changes_l1737_173709


namespace negation_of_false_l1737_173770

theorem negation_of_false (p q : Prop) : p ∧ ¬q → ¬q := by
  sorry

end negation_of_false_l1737_173770


namespace interest_rate_is_twelve_percent_l1737_173763

/-- Calculate the interest rate given principal, time, and interest amount -/
def calculate_interest_rate (principal : ℕ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest * 100) / (principal * time)

/-- Theorem: The interest rate is 12% given the problem conditions -/
theorem interest_rate_is_twelve_percent (principal : ℕ) (time : ℕ) (interest : ℕ)
  (h1 : principal = 9200)
  (h2 : time = 3)
  (h3 : interest = principal - 5888) :
  calculate_interest_rate principal time interest = 12 := by
  sorry

#eval calculate_interest_rate 9200 3 (9200 - 5888)

end interest_rate_is_twelve_percent_l1737_173763


namespace felipe_construction_time_l1737_173743

theorem felipe_construction_time :
  ∀ (felipe_time emilio_time : ℝ) (felipe_break emilio_break : ℝ),
    felipe_time + emilio_time = 7.5 * 12 →
    felipe_time = emilio_time / 2 →
    felipe_break = 6 →
    emilio_break = 2 * felipe_break →
    felipe_time + felipe_break = 36 :=
by
  sorry

end felipe_construction_time_l1737_173743


namespace projection_a_on_b_l1737_173742

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, -4)

theorem projection_a_on_b : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -6/5 := by sorry

end projection_a_on_b_l1737_173742


namespace hexagon_segment_probability_l1737_173755

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of different diagonal lengths in a regular hexagon -/
def num_diagonal_lengths : ℕ := 3

/-- The number of diagonals of each length in a regular hexagon -/
def diagonals_per_length : ℕ := num_diagonals / num_diagonal_lengths

theorem hexagon_segment_probability : 
  (num_sides * (num_sides - 1) + num_diagonals * (diagonals_per_length - 1)) / 
  (total_segments * (total_segments - 1)) = 11 / 35 := by
sorry

end hexagon_segment_probability_l1737_173755


namespace rectangle_area_error_percentage_l1737_173768

/-- Given a rectangle where one side is measured 16% in excess and the other side is measured 5% in deficit, 
    the error percentage in the calculated area is 10.2%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L_positive : L > 0) (W_positive : W > 0) : 
  let actual_area := L * W
  let measured_length := L * (1 + 16/100)
  let measured_width := W * (1 - 5/100)
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 10.2 := by
sorry

end rectangle_area_error_percentage_l1737_173768


namespace sum_of_common_ratios_is_three_l1737_173773

-- Define the common ratios
variable (p r : ℝ)

-- Define the first term of both sequences
variable (k : ℝ)

-- Define the geometric sequences
def a (n : ℕ) : ℝ := k * p^n
def b (n : ℕ) : ℝ := k * r^n

-- State the theorem
theorem sum_of_common_ratios_is_three 
  (h1 : p ≠ 1) 
  (h2 : r ≠ 1) 
  (h3 : p ≠ r) 
  (h4 : k ≠ 0) 
  (h5 : a 4 - b 4 = 4 * (a 2 - b 2)) : 
  p + r = 3 := by
sorry

end sum_of_common_ratios_is_three_l1737_173773


namespace min_value_xyz_min_value_exact_l1737_173710

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → x^3 * y^2 * z ≤ a^3 * b^2 * c :=
by sorry

theorem min_value_exact (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 9 ∧ a^3 * b^2 * c = 1/46656 :=
by sorry

end min_value_xyz_min_value_exact_l1737_173710


namespace sugar_solution_replacement_l1737_173758

/-- Represents a sugar solution with a total weight and sugar percentage -/
structure SugarSolution where
  totalWeight : ℝ
  sugarPercentage : ℝ

/-- Represents the mixing of two sugar solutions -/
def mixSolutions (original : SugarSolution) (replacement : SugarSolution) (replacementFraction : ℝ) : SugarSolution :=
  { totalWeight := original.totalWeight,
    sugarPercentage := 
      (1 - replacementFraction) * original.sugarPercentage + 
      replacementFraction * replacement.sugarPercentage }

theorem sugar_solution_replacement (original : SugarSolution) (replacement : SugarSolution) :
  original.sugarPercentage = 12 →
  (mixSolutions original replacement (1/4)).sugarPercentage = 16 →
  replacement.sugarPercentage = 28 := by
  sorry

end sugar_solution_replacement_l1737_173758


namespace unique_solution_l1737_173726

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0 ∧
  x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0

/-- The theorem stating that (2, 1) is the only solution to the system -/
theorem unique_solution :
  ∀ x y : ℝ, system x y ↔ x = 2 ∧ y = 1 := by
  sorry

end unique_solution_l1737_173726


namespace rod_length_relation_l1737_173765

/-- Two homogeneous rods with equal cross-sectional areas, different densities, and different
    coefficients of expansion are welded together. The system's center of gravity remains
    unchanged despite thermal expansion. -/
theorem rod_length_relation (l₁ l₂ d₁ d₂ α₁ α₂ : ℝ) 
    (h₁ : l₁ > 0) (h₂ : l₂ > 0) (h₃ : d₁ > 0) (h₄ : d₂ > 0) (h₅ : α₁ > 0) (h₆ : α₂ > 0) :
    (l₁ / l₂)^2 = (d₂ * α₂) / (d₁ * α₁) := by
  sorry

end rod_length_relation_l1737_173765


namespace inscribed_circumscribed_inequality_l1737_173722

/-- A polygon inscribed in one circle and circumscribed around another -/
structure InscribedCircumscribedPolygon where
  /-- Area of the inscribing circle -/
  A : ℝ
  /-- Area of the polygon -/
  B : ℝ
  /-- Area of the circumscribed circle -/
  C : ℝ
  /-- The inscribing circle has positive area -/
  hA : 0 < A
  /-- The polygon has positive area -/
  hB : 0 < B
  /-- The circumscribed circle has positive area -/
  hC : 0 < C
  /-- The polygon's area is less than or equal to the inscribing circle's area -/
  hAB : B ≤ A
  /-- The circumscribed circle's area is less than or equal to the polygon's area -/
  hBC : C ≤ B

/-- The inequality holds for any inscribed-circumscribed polygon configuration -/
theorem inscribed_circumscribed_inequality (p : InscribedCircumscribedPolygon) : 2 * p.B ≤ p.A + p.C := by
  sorry

end inscribed_circumscribed_inequality_l1737_173722


namespace line_y_intercept_l1737_173754

/-- A line with slope -3 and x-intercept (4,0) has y-intercept (0,12) -/
theorem line_y_intercept (f : ℝ → ℝ) (h1 : ∀ x y, f y - f x = -3 * (y - x)) 
  (h2 : f 4 = 0) : f 0 = 12 := by
  sorry

end line_y_intercept_l1737_173754


namespace brenda_stones_count_brenda_bought_36_stones_l1737_173701

theorem brenda_stones_count : ℕ → ℕ → ℕ
  | num_bracelets, stones_per_bracelet => 
    num_bracelets * stones_per_bracelet

theorem brenda_bought_36_stones 
  (num_bracelets : ℕ) 
  (stones_per_bracelet : ℕ) 
  (h1 : num_bracelets = 3) 
  (h2 : stones_per_bracelet = 12) : 
  brenda_stones_count num_bracelets stones_per_bracelet = 36 := by
  sorry

end brenda_stones_count_brenda_bought_36_stones_l1737_173701


namespace imaginary_cube_plus_one_l1737_173702

theorem imaginary_cube_plus_one (i : ℂ) : i^2 = -1 → 1 + i^3 = 1 - i := by
  sorry

end imaginary_cube_plus_one_l1737_173702


namespace percentage_of_36_l1737_173790

theorem percentage_of_36 : (33 + 1 / 3 : ℚ) / 100 * 36 = 12 := by sorry

end percentage_of_36_l1737_173790


namespace range_of_a_inequality_proof_l1737_173719

-- Define the function f
def f (x : ℝ) : ℝ := 3 * |x - 1| + |3 * x + 7|

-- Part 1
theorem range_of_a (a : ℝ) : 
  (∀ x, f x ≥ a^2 - 3*a) → -2 ≤ a ∧ a ≤ 5 := by sorry

-- Part 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  ∀ x, Real.sqrt (a + 1) + Real.sqrt (b + 1) ≤ Real.sqrt (f x) := by sorry

end range_of_a_inequality_proof_l1737_173719


namespace daily_apple_harvest_l1737_173708

/-- The number of sections in the apple orchard -/
def num_sections : ℕ := 8

/-- The number of sacks of apples harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks of apples harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

theorem daily_apple_harvest :
  total_sacks = 360 :=
by sorry

end daily_apple_harvest_l1737_173708


namespace minimum_value_and_range_proof_l1737_173771

theorem minimum_value_and_range_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 9 ∧ (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1/a' + 4/b' ≥ min) ∧
    (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1/a₀ + 4/b₀ = min)) ∧
  (∀ x : ℝ, (1/a + 4/b ≥ |2*x - 1| - |x + 1|) → -7 ≤ x ∧ x ≤ 11) :=
by sorry

end minimum_value_and_range_proof_l1737_173771


namespace product_of_three_integers_l1737_173772

theorem product_of_three_integers : (-3 : ℤ) * (-4 : ℤ) * (-1 : ℤ) = -12 := by
  sorry

end product_of_three_integers_l1737_173772


namespace triangle_theorem_l1737_173792

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

theorem triangle_theorem (t : Triangle) 
  (h : t.b * Real.cos t.A + Real.sqrt 3 * t.b * Real.sin t.A - t.c - t.a = 0) :
  t.B = π / 3 ∧ 
  (t.b = Real.sqrt 3 → ∀ (a c : ℝ), a + c ≤ 2 * Real.sqrt 3) :=
by sorry

end triangle_theorem_l1737_173792


namespace chocolate_bar_eating_ways_l1737_173724

/-- Represents a chocolate bar of size m × n -/
structure ChocolateBar (m n : ℕ) where
  size : Fin m × Fin n → Bool

/-- Represents the state of eating a chocolate bar -/
structure EatingState (m n : ℕ) where
  bar : ChocolateBar m n
  eaten : Fin m × Fin n → Bool

/-- Checks if a piece can be eaten (has no more than two shared sides with uneaten pieces) -/
def canEat (state : EatingState m n) (pos : Fin m × Fin n) : Bool :=
  sorry

/-- Counts the number of ways to eat the chocolate bar -/
def countEatingWays (m n : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 6720 ways to eat a 2 × 4 chocolate bar -/
theorem chocolate_bar_eating_ways :
  countEatingWays 2 4 = 6720 :=
sorry

end chocolate_bar_eating_ways_l1737_173724


namespace opposite_of_negative_two_l1737_173729

theorem opposite_of_negative_two (a : ℝ) : a = -(- 2) → a = 2 := by
  sorry

end opposite_of_negative_two_l1737_173729


namespace system_solution_l1737_173717

theorem system_solution (a b c : ℂ) : 
  (a^2 + a*b + c = 0 ∧ b^2 + b*c + a = 0 ∧ c^2 + c*a + b = 0) → 
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2)) := by
  sorry

end system_solution_l1737_173717


namespace complex_sum_powers_l1737_173739

theorem complex_sum_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := by
  sorry

end complex_sum_powers_l1737_173739


namespace cube_face_sum_l1737_173757

theorem cube_face_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f = 1089) → 
  (a + b + c + d + e + f = 31) := by
sorry

end cube_face_sum_l1737_173757


namespace valid_numbers_l1737_173774

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (p : ℕ),
    n = m + 10^k * a + 10^(k+2) * p ∧
    0 ≤ a ∧ a < 100 ∧
    m < 10^k ∧
    n = 87 * (m + 10^k * p) ∧
    n ≥ 10^99 ∧ n < 10^100

theorem valid_numbers :
  {n : ℕ | is_valid_number n} =
    {435 * 10^97, 1305 * 10^96, 2175 * 10^96, 3045 * 10^96} :=
by sorry

end valid_numbers_l1737_173774


namespace total_puff_pastries_made_l1737_173744

/-- Theorem: Calculating total puff pastries made by volunteers -/
theorem total_puff_pastries_made
  (num_volunteers : ℕ)
  (trays_per_batch : ℕ)
  (pastries_per_tray : ℕ)
  (h1 : num_volunteers = 1000)
  (h2 : trays_per_batch = 8)
  (h3 : pastries_per_tray = 25) :
  num_volunteers * trays_per_batch * pastries_per_tray = 200000 :=
by sorry

end total_puff_pastries_made_l1737_173744


namespace iron_volume_change_l1737_173716

/-- If the volume of iron reduces by 1/34 when solidifying, then the volume increases by 1/33 when melting back to its original state. -/
theorem iron_volume_change (V : ℝ) (V_block : ℝ) (h : V_block = V * (1 - 1/34)) :
  (V - V_block) / V_block = 1/33 := by
sorry

end iron_volume_change_l1737_173716


namespace least_subtraction_for_divisibility_problem_solution_l1737_173779

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 196713
  let d := 7
  let x := 6
  x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l1737_173779


namespace circle_area_sum_l1737_173748

/-- The sum of the areas of an infinite sequence of circles with decreasing radii -/
theorem circle_area_sum : 
  let r : ℕ → ℝ := λ n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (r n)^2
  (∑' n, area n) = 9*π/2 := by
  sorry

end circle_area_sum_l1737_173748


namespace granola_initial_price_l1737_173778

/-- Proves that the initial selling price per bag was $6.00 --/
theorem granola_initial_price (ingredient_cost : ℝ) (total_bags : ℕ) 
  (full_price_sold : ℕ) (discount_price : ℝ) (net_profit : ℝ) :
  ingredient_cost = 3 →
  total_bags = 20 →
  full_price_sold = 15 →
  discount_price = 4 →
  net_profit = 50 →
  ∃ (initial_price : ℝ), 
    initial_price * full_price_sold + discount_price * (total_bags - full_price_sold) - 
    ingredient_cost * total_bags = net_profit ∧
    initial_price = 6 := by
  sorry

#check granola_initial_price

end granola_initial_price_l1737_173778


namespace northton_capsule_depth_l1737_173781

theorem northton_capsule_depth (southton_depth : ℝ) (northton_offset : ℝ) : 
  southton_depth = 15 →
  northton_offset = 12 →
  (4 * southton_depth + northton_offset) = 72 :=
by
  sorry

end northton_capsule_depth_l1737_173781


namespace perpendicular_lines_parallel_l1737_173747

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end perpendicular_lines_parallel_l1737_173747


namespace equal_commissions_l1737_173767

/-- The list price of an item that satisfies the given conditions -/
def list_price : ℝ := 40

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Charlie's selling price -/
def charlie_price (x : ℝ) : ℝ := x - 20

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := 0.25 * bob_price x

/-- Charlie's commission -/
def charlie_commission (x : ℝ) : ℝ := 0.20 * charlie_price x

theorem equal_commissions :
  alice_commission list_price = bob_commission list_price ∧
  bob_commission list_price = charlie_commission list_price := by
  sorry

end equal_commissions_l1737_173767


namespace gift_distribution_ways_l1737_173700

theorem gift_distribution_ways (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (Nat.factorial n) / (Nat.factorial (n - k)) = 303600 := by
  sorry

end gift_distribution_ways_l1737_173700


namespace initial_tagged_fish_count_l1737_173706

/-- The number of fish initially tagged and returned to the pond -/
def initial_tagged_fish : ℕ := 50

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1250

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish in the second catch -/
def tagged_in_second_catch : ℕ := 2

theorem initial_tagged_fish_count :
  initial_tagged_fish = 50 := by sorry

end initial_tagged_fish_count_l1737_173706


namespace point_on_graph_l1737_173783

/-- The function f(x) = -3x + 3 -/
def f (x : ℝ) : ℝ := -3 * x + 3

/-- The point p = (-2, 9) -/
def p : ℝ × ℝ := (-2, 9)

/-- Theorem: The point p lies on the graph of f -/
theorem point_on_graph : f p.1 = p.2 := by sorry

end point_on_graph_l1737_173783


namespace not_divisible_by_49_l1737_173712

theorem not_divisible_by_49 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 3*n + 4 = 49*k := by
  sorry

end not_divisible_by_49_l1737_173712


namespace count_of_six_from_100_to_999_l1737_173718

/-- Count of digit 6 in a specific place (units, tens, or hundreds) for numbers from 100 to 999 -/
def count_digit_in_place (place : Nat) : Nat :=
  if place = 2 then 100 else 90

/-- Total count of digit 6 in all places for numbers from 100 to 999 -/
def total_count_of_six : Nat :=
  count_digit_in_place 0 + count_digit_in_place 1 + count_digit_in_place 2

/-- Theorem: The digit 6 appears 280 times when writing integers from 100 through 999 inclusive -/
theorem count_of_six_from_100_to_999 : total_count_of_six = 280 := by
  sorry

end count_of_six_from_100_to_999_l1737_173718


namespace complex_roots_on_circle_l1737_173741

theorem complex_roots_on_circle : ∃ (r : ℝ), r = 2 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z + 2)^6 = 64 * z^6 → Complex.abs (z - Complex.ofReal (2/3)) = r :=
sorry

end complex_roots_on_circle_l1737_173741


namespace interior_angles_sum_not_270_l1737_173723

theorem interior_angles_sum_not_270 (n : ℕ) (h : 3 ≤ n ∧ n ≤ 5) :
  (n - 2) * 180 ≠ 270 := by
  sorry

end interior_angles_sum_not_270_l1737_173723


namespace range_of_f_l1737_173720

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 3 :=
sorry

end range_of_f_l1737_173720


namespace friends_team_assignment_l1737_173727

theorem friends_team_assignment (n : ℕ) (k : ℕ) :
  (n = 8 ∧ k = 4) →
  (number_of_assignments : ℕ) →
  number_of_assignments = k^n :=
by
  sorry

end friends_team_assignment_l1737_173727


namespace integer_less_than_sqrt5_l1737_173795

theorem integer_less_than_sqrt5 : ∃ z : ℤ, |z| < Real.sqrt 5 := by
  sorry

end integer_less_than_sqrt5_l1737_173795


namespace imaginary_part_of_z_l1737_173740

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z := (1 + i^3) / (2 - i)
  Complex.im z = -1/5 := by sorry

end imaginary_part_of_z_l1737_173740


namespace boxes_needed_to_sell_l1737_173721

def total_chocolate_bars : ℕ := 710
def chocolate_bars_per_box : ℕ := 5

theorem boxes_needed_to_sell (total : ℕ) (per_box : ℕ) :
  total = total_chocolate_bars →
  per_box = chocolate_bars_per_box →
  total / per_box = 142 := by
  sorry

end boxes_needed_to_sell_l1737_173721


namespace closest_integer_to_sqrt_35_l1737_173746

theorem closest_integer_to_sqrt_35 :
  ∀ x : ℝ, x = Real.sqrt 35 → (5 < x ∧ x < 6) → ∀ n : ℤ, |x - 6| ≤ |x - n| :=
by sorry

end closest_integer_to_sqrt_35_l1737_173746


namespace max_k_value_l1737_173782

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1 - 2*m) ≥ k) → k ≤ 8) ∧ 
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1 - 2*m) ≥ k) :=
sorry

end max_k_value_l1737_173782


namespace water_evaporation_l1737_173749

/-- Given a bowl with 10 ounces of water, if 2% of the original amount evaporates
    over 50 days, then the amount of water evaporated each day is 0.04 ounces. -/
theorem water_evaporation (initial_water : ℝ) (days : ℕ) (evaporation_rate : ℝ) :
  initial_water = 10 →
  days = 50 →
  evaporation_rate = 0.02 →
  (initial_water * evaporation_rate) / days = 0.04 :=
by sorry

end water_evaporation_l1737_173749


namespace no_real_solution_log_equation_l1737_173736

theorem no_real_solution_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 4) + Real.log (x - 2) = Real.log (x^2 - 6*x + 8)) ∧ 
  (x + 4 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 6*x + 8 > 0) :=
sorry

end no_real_solution_log_equation_l1737_173736


namespace rectangular_strip_area_l1737_173732

theorem rectangular_strip_area (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43 →
  a = 1 ∧ b + c = 22 := by
sorry

end rectangular_strip_area_l1737_173732


namespace inequality_proof_l1737_173797

theorem inequality_proof (m : ℕ+) (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^(m : ℕ) / ((1 + y) * (1 + z))) + 
  (y^(m : ℕ) / ((1 + x) * (1 + z))) + 
  (z^(m : ℕ) / ((1 + x) * (1 + y))) ≥ 3/4 := by
sorry

end inequality_proof_l1737_173797


namespace kwik_e_tax_revenue_l1737_173751

/-- Calculates the total revenue for Kwik-e-Tax Center given the prices and number of returns sold --/
def total_revenue (federal_price state_price quarterly_price : ℕ) 
                  (federal_sold state_sold quarterly_sold : ℕ) : ℕ :=
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold

/-- Theorem stating that the total revenue for the given scenario is $4400 --/
theorem kwik_e_tax_revenue : 
  total_revenue 50 30 80 60 20 10 = 4400 := by
sorry

end kwik_e_tax_revenue_l1737_173751


namespace inequality_solution_and_function_property_l1737_173730

def f (x : ℝ) := |x - 2|

theorem inequality_solution_and_function_property :
  (∀ x : ℝ, (|x - 2| + |x| ≤ 4) ↔ (x ∈ Set.Icc (-1) 3)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a ≠ b → a * f b + b * f a ≥ 2 * |a - b|) :=
sorry

end inequality_solution_and_function_property_l1737_173730


namespace lcm_factor_14_l1737_173775

theorem lcm_factor_14 (A B : ℕ+) (h1 : Nat.gcd A B = 16) (h2 : A = 224) :
  ∃ (X Y : ℕ+), Nat.lcm A B = 16 * X * Y ∧ (X = 14 ∨ Y = 14) := by
sorry

end lcm_factor_14_l1737_173775


namespace unique_prime_between_squares_l1737_173738

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 4 ∧ 
  ∃ m : ℕ, p + 7 = (n + 1)^2 ∧ 
  p = 29 := by
sorry

end unique_prime_between_squares_l1737_173738


namespace decimal_521_equals_octal_1011_l1737_173785

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid octal number -/
def is_valid_octal (l : List ℕ) : Prop :=
  l.all (λ d => d < 8)

theorem decimal_521_equals_octal_1011 :
  decimal_to_octal 521 = [1, 0, 1, 1] ∧ is_valid_octal [1, 0, 1, 1] :=
by sorry

end decimal_521_equals_octal_1011_l1737_173785


namespace square_equals_self_implies_zero_or_one_l1737_173752

theorem square_equals_self_implies_zero_or_one (a : ℝ) : a^2 = a → a = 0 ∨ a = 1 := by
  sorry

end square_equals_self_implies_zero_or_one_l1737_173752


namespace divisibility_by_24_l1737_173745

theorem divisibility_by_24 (n : Nat) : n ≤ 9 → (712 * 10 + n) % 24 = 0 ↔ n = 8 := by
  sorry

end divisibility_by_24_l1737_173745


namespace fibonacci_periodicity_last_digit_2020th_fibonacci_l1737_173769

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

def last_digit (n : ℕ) : ℕ := n % 10

theorem fibonacci_periodicity (n : ℕ) : last_digit (fibonacci n) = last_digit (fibonacci (n % 60)) := by sorry

theorem last_digit_2020th_fibonacci : last_digit (fibonacci 2020) = 0 := by sorry

end fibonacci_periodicity_last_digit_2020th_fibonacci_l1737_173769


namespace cubic_difference_equals_2011_l1737_173793

theorem cubic_difference_equals_2011 (x y : ℕ+) (h : x.val^2 - y.val^2 = 53) :
  x.val^3 - y.val^3 - 2 * (x.val + y.val) + 10 = 2011 := by
  sorry

end cubic_difference_equals_2011_l1737_173793


namespace polynomial_remainder_l1737_173799

def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 3

theorem polynomial_remainder : p 2 = -1 := by
  sorry

end polynomial_remainder_l1737_173799


namespace sum_of_areas_equals_100_l1737_173766

-- Define the circle radius
def circle_radius : ℝ := 5

-- Define the maximum rectangle inscribed in the circle
def max_rectangle_area (r : ℝ) : ℝ := 2 * r^2

-- Define the maximum parallelogram circumscribed around the circle
def max_parallelogram_area (r : ℝ) : ℝ := 4 * r^2

-- Theorem statement
theorem sum_of_areas_equals_100 :
  max_rectangle_area circle_radius + max_parallelogram_area circle_radius = 100 := by
  sorry

#eval max_rectangle_area circle_radius + max_parallelogram_area circle_radius

end sum_of_areas_equals_100_l1737_173766


namespace logarithm_expression_equality_l1737_173715

theorem logarithm_expression_equality : 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + 2^(Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end logarithm_expression_equality_l1737_173715


namespace morgans_list_count_l1737_173759

theorem morgans_list_count : ∃ n : ℕ, n = 871 ∧ 
  n = (Finset.range (27000 / 30 + 1) \ Finset.range (900 / 30)).card := by
  sorry

end morgans_list_count_l1737_173759


namespace solution_set_equivalence_l1737_173731

theorem solution_set_equivalence :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end solution_set_equivalence_l1737_173731


namespace division_problem_l1737_173705

theorem division_problem (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient → 
  divisor = 5 * remainder → 
  remainder = 46 → 
  divisor * quotient + remainder = 5336 := by
sorry

end division_problem_l1737_173705


namespace all_triangles_in_S_are_similar_l1737_173798

-- Define a structure for triangles in set S
structure TriangleS where
  A : Real
  B : Real
  C : Real
  tan_A_pos_int : ℕ+
  tan_B_pos_int : ℕ+
  tan_C_pos_int : ℕ+
  angle_sum : A + B + C = Real.pi
  tan_sum_identity : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C

-- Define similarity for triangles in S
def similar (t1 t2 : TriangleS) : Prop :=
  ∃ (k : Real), k > 0 ∧
    t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- State the theorem
theorem all_triangles_in_S_are_similar (t1 t2 : TriangleS) :
  similar t1 t2 := by
  sorry

end all_triangles_in_S_are_similar_l1737_173798


namespace koh_nh4i_reaction_l1737_173735

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactants : List String
  products : List String
  ratio : Nat

/-- Represents the state of a chemical system -/
structure ChemicalSystem where
  compounds : List String
  moles : List ℚ

/-- Calculates the moles of products formed and remaining reactants -/
def reactComplete (reaction : ChemicalReaction) (initial : ChemicalSystem) : ChemicalSystem :=
  sorry

theorem koh_nh4i_reaction 
  (reaction : ChemicalReaction)
  (initial : ChemicalSystem)
  (h_reaction : reaction = 
    { reactants := ["KOH", "NH4I"]
    , products := ["KI", "NH3", "H2O"]
    , ratio := 1 })
  (h_initial : initial = 
    { compounds := ["KOH", "NH4I"]
    , moles := [3, 3] })
  : 
  let final := reactComplete reaction initial
  (final.compounds = ["KI", "NH3", "H2O", "KOH", "NH4I"] ∧
   final.moles = [3, 3, 3, 0, 0]) :=
by sorry

end koh_nh4i_reaction_l1737_173735


namespace stratified_sample_size_l1737_173737

-- Define the ratio of quantities for models A, B, and C
def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 4

-- Define the number of units of model A in the sample
def units_A : ℕ := 16

-- Define the total sample size
def sample_size : ℕ := units_A + (ratio_B * units_A / ratio_A) + (ratio_C * units_A / ratio_A)

-- Theorem statement
theorem stratified_sample_size :
  sample_size = 72 :=
sorry

end stratified_sample_size_l1737_173737


namespace D_72_l1737_173733

/-- D(n) is the number of ways to write n as a product of integers greater than 1, 
    considering the order of factors. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(72) = 97 -/
theorem D_72 : D 72 = 97 := by sorry

end D_72_l1737_173733


namespace folding_crease_set_l1737_173713

/-- Given a circle with center O(0,0) and radius R, and a point A(a,0) inside the circle,
    the set of all points P(x,y) that are equidistant from A and any point A' on the circumference
    of the circle satisfies the given inequality. -/
theorem folding_crease_set (R a x y : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < R) :
  (x - a/2)^2 / (R/2)^2 + y^2 / ((R/2)^2 - (a/2)^2) ≥ 1 :=
sorry

end folding_crease_set_l1737_173713


namespace profit_calculation_l1737_173787

-- Define the buy rate
def buy_rate : ℚ := 15 / 4

-- Define the sell rate
def sell_rate : ℚ := 30 / 6

-- Define the target profit
def target_profit : ℚ := 200

-- Define the number of oranges to be sold
def oranges_to_sell : ℕ := 160

-- Theorem statement
theorem profit_calculation :
  (oranges_to_sell : ℚ) * (sell_rate - buy_rate) = target_profit :=
by sorry

end profit_calculation_l1737_173787


namespace dragon_can_be_defeated_l1737_173762

/-- Represents the possible number of heads a warrior can chop off in one strike -/
inductive Strike
  | thirtythree
  | twentyone
  | seventeen
  | one

/-- Represents the state of the dragon -/
structure DragonState where
  heads : ℕ

/-- Applies a strike to the dragon state -/
def applyStrike (s : Strike) (d : DragonState) : DragonState :=
  match s with
  | Strike.thirtythree => ⟨d.heads + 48 - 33⟩
  | Strike.twentyone => ⟨d.heads - 21⟩
  | Strike.seventeen => ⟨d.heads + 14 - 17⟩
  | Strike.one => ⟨d.heads + 349 - 1⟩

/-- Represents a sequence of strikes -/
def StrikeSequence := List Strike

/-- Applies a sequence of strikes to the dragon state -/
def applySequence (seq : StrikeSequence) (d : DragonState) : DragonState :=
  seq.foldl (fun state strike => applyStrike strike state) d

/-- The theorem stating that the dragon can be defeated -/
theorem dragon_can_be_defeated : 
  ∃ (seq : StrikeSequence), (applySequence seq ⟨2000⟩).heads = 0 := by
  sorry

end dragon_can_be_defeated_l1737_173762


namespace age_ratio_proof_l1737_173794

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 42 →  -- The total of the ages of a, b, and c is 42
  b = 16 →  -- b is 16 years old
  b = 2 * c  -- The ratio of b's age to c's age is 2:1
  := by sorry

end age_ratio_proof_l1737_173794


namespace smallest_possible_b_l1737_173711

theorem smallest_possible_b (a b c : ℝ) : 
  1 < a → a < b → c = 2 → 
  (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) →
  (¬ ((1/b) + (1/a) > c ∧ (1/b) + c > (1/a) ∧ (1/a) + c > (1/b))) →
  b ≥ 2 ∧ ∀ x, (x > 1 ∧ x < b → x ≥ a) → b = 2 :=
by sorry

end smallest_possible_b_l1737_173711


namespace root_product_expression_l1737_173789

theorem root_product_expression (p q r s : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + r = 0) → 
  (β^2 + p*β + r = 0) → 
  (γ^2 + q*γ + s = 0) → 
  (δ^2 + q*δ + s = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = (p-q)^4 * s^2 + 2*(p-q)^3 * s * (r-s) + (p-q)^2 * (r-s)^2 := by
sorry

end root_product_expression_l1737_173789


namespace uncle_welly_roses_l1737_173734

/-- Proves that Uncle Welly planted 20 more roses yesterday compared to two days ago -/
theorem uncle_welly_roses : 
  ∀ (roses_two_days_ago roses_yesterday roses_today : ℕ),
  roses_two_days_ago = 50 →
  roses_today = 2 * roses_two_days_ago →
  roses_yesterday > roses_two_days_ago →
  roses_two_days_ago + roses_yesterday + roses_today = 220 →
  roses_yesterday - roses_two_days_ago = 20 := by
sorry


end uncle_welly_roses_l1737_173734


namespace charles_earnings_correct_l1737_173756

/-- Calculates Charles' earnings after tax deduction based on his housesitting and dog walking activities. -/
def charles_earnings : ℝ :=
  let housesitting_rate : ℝ := 15
  let labrador_rate : ℝ := 22
  let golden_retriever_rate : ℝ := 25
  let german_shepherd_rate : ℝ := 30
  let housesitting_hours : ℝ := 10
  let labrador_hours : ℝ := 3
  let golden_retriever_hours : ℝ := 2
  let german_shepherd_hours : ℝ := 1.5
  let tax_rate : ℝ := 0.1

  let total_before_tax : ℝ := 
    housesitting_rate * housesitting_hours +
    labrador_rate * labrador_hours * 2 +
    golden_retriever_rate * golden_retriever_hours +
    german_shepherd_rate * german_shepherd_hours

  total_before_tax * (1 - tax_rate)

theorem charles_earnings_correct : charles_earnings = 339.30 := by
  sorry

end charles_earnings_correct_l1737_173756


namespace tan_beta_value_l1737_173761

theorem tan_beta_value (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β / 2) = 1 / 3) :
  Real.tan β = 7 / 24 := by
  sorry

end tan_beta_value_l1737_173761


namespace f_2015_value_l1737_173764

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x, f (x + 1) = f (1 - x)) ∧
  (f 1 = 5)

theorem f_2015_value (f : ℝ → ℝ) (h : f_properties f) : f 2015 = -5 := by
  sorry

end f_2015_value_l1737_173764


namespace cube_sum_geq_sqrt_product_square_sum_l1737_173750

theorem cube_sum_geq_sqrt_product_square_sum {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := by
  sorry

end cube_sum_geq_sqrt_product_square_sum_l1737_173750


namespace least_n_modulo_121_l1737_173788

theorem least_n_modulo_121 : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(25^m + 16^m) % 121 = 1) ∧ (25^n + 16^n) % 121 = 1 :=
by
  use 32
  sorry

end least_n_modulo_121_l1737_173788


namespace inequality_proof_l1737_173784

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
sorry

end inequality_proof_l1737_173784
