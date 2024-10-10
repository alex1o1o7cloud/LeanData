import Mathlib

namespace sqrt_sum_eq_three_l1400_140076

theorem sqrt_sum_eq_three (a : ℝ) (h : a + 1/a = 7) : 
  Real.sqrt a + 1 / Real.sqrt a = 3 := by
sorry

end sqrt_sum_eq_three_l1400_140076


namespace complex_division_result_l1400_140089

/-- Given that z = a^2 - 1 + (1 + a)i where a ∈ ℝ is a purely imaginary number,
    prove that z / (2 + i) = 2/5 + 4/5 * i -/
theorem complex_division_result (a : ℝ) (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = a^2 - 1 + (1 + a) * i →
  z.re = 0 →
  z / (2 + i) = 2/5 + 4/5 * i :=
by sorry

end complex_division_result_l1400_140089


namespace eighth_term_of_sequence_l1400_140032

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem eighth_term_of_sequence 
  (a : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a d 4 = 23) 
  (h2 : arithmetic_sequence a d 6 = 47) : 
  arithmetic_sequence a d 8 = 71 := by
sorry

end eighth_term_of_sequence_l1400_140032


namespace problem_1_problem_2_l1400_140049

-- Define the ☆ operation
def star (m n : ℝ) : ℝ := m^2 - m*n + n

-- Theorem statements
theorem problem_1 : star 3 4 = 1 := by sorry

theorem problem_2 : star (-1) (star 2 (-3)) = 15 := by sorry

end problem_1_problem_2_l1400_140049


namespace fraction_calculation_l1400_140000

theorem fraction_calculation : 
  (((1 / 6 : ℚ) - (1 / 8 : ℚ) + (1 / 9 : ℚ)) / ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 5 : ℚ))) * 3 = 55 / 34 := by
  sorry

end fraction_calculation_l1400_140000


namespace stratified_sampling_female_count_l1400_140068

/-- Calculates the number of females in a population given stratified sampling data -/
theorem stratified_sampling_female_count 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (females_in_sample : ℕ) 
  (total_population_pos : 0 < total_population)
  (sample_size_pos : 0 < sample_size)
  (sample_size_le_total : sample_size ≤ total_population)
  (females_in_sample_le_sample : females_in_sample ≤ sample_size) :
  let females_in_population : ℕ := (females_in_sample * total_population) / sample_size
  females_in_population = 760 ∧ females_in_population ≤ total_population :=
by sorry

end stratified_sampling_female_count_l1400_140068


namespace parabola_focus_point_slope_l1400_140006

/-- The slope of a line between the focus of a parabola and a point on the parabola -/
theorem parabola_focus_point_slope (x y : ℝ) :
  y^2 = 4*x →  -- parabola equation
  x > 0 →  -- point is in the fourth quadrant
  y < 0 →  -- point is in the fourth quadrant
  x + 1 = 5 →  -- distance from point to directrix is 5
  (y - 0) / (x - 1) = -4/3 :=  -- slope of line AF
by sorry

end parabola_focus_point_slope_l1400_140006


namespace solve_potato_problem_l1400_140090

def potatoesProblem (initialPotatoes : ℕ) (ginaAmount : ℕ) : Prop :=
  let tomAmount := 2 * ginaAmount
  let anneAmount := tomAmount / 3
  let remainingPotatoes := initialPotatoes - (ginaAmount + tomAmount + anneAmount)
  remainingPotatoes = 47

theorem solve_potato_problem :
  potatoesProblem 300 69 := by
  sorry

end solve_potato_problem_l1400_140090


namespace nabla_calculation_l1400_140024

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_calculation_l1400_140024


namespace prism_volume_l1400_140005

/-- Given a right rectangular prism with face areas 30 cm², 50 cm², and 75 cm², 
    its volume is 335 cm³. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 50) 
  (h3 : b * c = 75) : 
  a * b * c = 335 := by
  sorry

end prism_volume_l1400_140005


namespace increasing_subsequence_exists_l1400_140067

/-- Given a sequence of 2^n positive integers where each element is at most its index,
    there exists a monotonically increasing subsequence of length n+1. -/
theorem increasing_subsequence_exists (n : ℕ) (a : Fin (2^n) → ℕ)
  (h : ∀ k : Fin (2^n), a k ≤ k.val + 1) :
  ∃ (s : Fin (n + 1) → Fin (2^n)), Monotone (a ∘ s) :=
sorry

end increasing_subsequence_exists_l1400_140067


namespace arc_length_of_sector_l1400_140046

/-- The arc length of a sector with radius 8 cm and central angle 45° is 2π cm. -/
theorem arc_length_of_sector (r : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  r = 8 → θ_deg = 45 → l = r * (θ_deg * π / 180) → l = 2 * π := by
  sorry

end arc_length_of_sector_l1400_140046


namespace king_spade_then_spade_probability_l1400_140094

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (spades : Nat)
  (king_of_spades : Nat)

/-- The probability of drawing a King of Spades followed by any Spade from a standard 52-card deck -/
def probability_king_spade_then_spade (d : Deck) : Rat :=
  (d.king_of_spades : Rat) / d.total_cards * (d.spades - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a King of Spades followed by any Spade 
    from a standard 52-card deck is 1/221 -/
theorem king_spade_then_spade_probability :
  probability_king_spade_then_spade ⟨52, 13, 1⟩ = 1 / 221 := by
  sorry

end king_spade_then_spade_probability_l1400_140094


namespace words_with_consonant_count_l1400_140017

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels -/
def all_vowel_words : ℕ := vowel_count ^ word_length

/-- The number of words with at least one consonant -/
def words_with_consonant : ℕ := total_words - all_vowel_words

theorem words_with_consonant_count : words_with_consonant = 7744 := by
  sorry

end words_with_consonant_count_l1400_140017


namespace no_solution_mod_five_l1400_140044

theorem no_solution_mod_five : ¬∃ (n : ℕ), n^2 % 5 = 1 ∧ n^3 % 5 = 3 := by
  sorry

end no_solution_mod_five_l1400_140044


namespace doughnut_cost_calculation_l1400_140061

/-- Calculates the total cost of doughnuts for a class -/
theorem doughnut_cost_calculation (total_students : ℕ) 
  (chocolate_lovers : ℕ) (glazed_lovers : ℕ) 
  (chocolate_cost : ℕ) (glazed_cost : ℕ) : 
  total_students = 25 →
  chocolate_lovers = 10 →
  glazed_lovers = 15 →
  chocolate_cost = 2 →
  glazed_cost = 1 →
  chocolate_lovers * chocolate_cost + glazed_lovers * glazed_cost = 35 :=
by
  sorry

#check doughnut_cost_calculation

end doughnut_cost_calculation_l1400_140061


namespace inverse_f_l1400_140063

/-- Given a function f: ℝ → ℝ satisfying f(4) = 3 and f(2x) = 2f(x) + 1 for all x,
    prove that f(128) = 127 -/
theorem inverse_f (f : ℝ → ℝ) (h1 : f 4 = 3) (h2 : ∀ x, f (2 * x) = 2 * f x + 1) :
  f 128 = 127 := by sorry

end inverse_f_l1400_140063


namespace max_lines_theorem_l1400_140093

/-- Given n points on a plane where no three are collinear, 
    this function returns the maximum number of lines that can be drawn 
    through pairs of points without forming a triangle with vertices 
    among the given points. -/
def max_lines_without_triangle (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 4
  else
    (n^2 - 1) / 4

/-- Theorem stating the maximum number of lines that can be drawn 
    through pairs of points without forming a triangle, 
    given n points on a plane where no three are collinear and n ≥ 3. -/
theorem max_lines_theorem (n : ℕ) (h : n ≥ 3) :
  max_lines_without_triangle n = 
    if n % 2 = 0 then
      n^2 / 4
    else
      (n^2 - 1) / 4 := by
  sorry

end max_lines_theorem_l1400_140093


namespace milkshake_cost_calculation_l1400_140097

/-- The cost of a milkshake given initial money, cupcake spending fraction, and remaining money --/
def milkshake_cost (initial : ℚ) (cupcake_fraction : ℚ) (remaining : ℚ) : ℚ :=
  initial - initial * cupcake_fraction - remaining

theorem milkshake_cost_calculation (initial : ℚ) (cupcake_fraction : ℚ) (remaining : ℚ) 
  (h1 : initial = 10)
  (h2 : cupcake_fraction = 1/5)
  (h3 : remaining = 3) :
  milkshake_cost initial cupcake_fraction remaining = 5 := by
  sorry

#eval milkshake_cost 10 (1/5) 3

end milkshake_cost_calculation_l1400_140097


namespace printer_pages_theorem_l1400_140072

/-- Represents a printer with specific crumpling and blurring patterns -/
structure Printer where
  crumple_interval : Nat
  blur_interval : Nat

/-- Calculates the number of pages that are neither crumpled nor blurred -/
def good_pages (p : Printer) (total : Nat) : Nat :=
  total - (total / p.crumple_interval + total / p.blur_interval - total / (Nat.lcm p.crumple_interval p.blur_interval))

/-- Theorem: For a printer that crumples every 7th page and blurs every 3rd page,
    if 24 pages are neither crumpled nor blurred, then 42 pages were printed in total -/
theorem printer_pages_theorem (p : Printer) (h1 : p.crumple_interval = 7) (h2 : p.blur_interval = 3) :
  good_pages p 42 = 24 := by
  sorry

#eval good_pages ⟨7, 3⟩ 42  -- Should output 24

end printer_pages_theorem_l1400_140072


namespace equation_one_solutions_equation_two_solutions_l1400_140041

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1 :=
sorry

end equation_one_solutions_equation_two_solutions_l1400_140041


namespace probability_diff_two_meters_l1400_140034

def bamboo_lengths : Finset ℕ := {1, 2, 3, 4}

def valid_pairs : Finset (ℕ × ℕ) :=
  {(1, 3), (3, 1), (2, 4), (4, 2)}

def total_pairs : Finset (ℕ × ℕ) :=
  bamboo_lengths.product bamboo_lengths

theorem probability_diff_two_meters :
  (valid_pairs.card : ℚ) / total_pairs.card = 1 / 3 := by
  sorry

end probability_diff_two_meters_l1400_140034


namespace proposition_p_and_q_true_l1400_140013

theorem proposition_p_and_q_true : 
  (∃ α : ℝ, Real.cos (π - α) = Real.cos α) ∧ (∀ x : ℝ, x^2 + 1 > 0) := by
  sorry

end proposition_p_and_q_true_l1400_140013


namespace inequality_solution_l1400_140091

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, ((x - p) * (x - q)) / (x - r) ≥ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2))
  (h2 : p < q) : 
  p + 2*q + 3*r = 78 := by
sorry

end inequality_solution_l1400_140091


namespace arrange_13_blue_5_red_l1400_140099

/-- The number of ways to arrange blue and red balls with constraints -/
def arrange_balls (blue_balls red_balls : ℕ) : ℕ :=
  Nat.choose (blue_balls - red_balls + 1 + red_balls) (red_balls + 1)

/-- Theorem: Arranging 13 blue balls and 5 red balls with constraints yields 2002 ways -/
theorem arrange_13_blue_5_red :
  arrange_balls 13 5 = 2002 := by
  sorry

#eval arrange_balls 13 5

end arrange_13_blue_5_red_l1400_140099


namespace unique_prime_between_30_and_40_with_remainder_4_mod_9_l1400_140042

theorem unique_prime_between_30_and_40_with_remainder_4_mod_9 :
  ∃! n : ℕ, 30 < n ∧ n < 40 ∧ Prime n ∧ n % 9 = 4 :=
by
  sorry

end unique_prime_between_30_and_40_with_remainder_4_mod_9_l1400_140042


namespace no_integer_solutions_to_3x2_plus_7y2_eq_z4_l1400_140019

theorem no_integer_solutions_to_3x2_plus_7y2_eq_z4 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 7 * y^2 = z^4 := by
  sorry

end no_integer_solutions_to_3x2_plus_7y2_eq_z4_l1400_140019


namespace fraction_equality_l1400_140098

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (4*x - y) / (3*x + 2*y) = 3) : 
  (3*x - 2*y) / (4*x + y) = 31/23 := by
sorry

end fraction_equality_l1400_140098


namespace jinx_hak_not_flog_l1400_140022

-- Define the sets
variable (U : Type) -- Universe set
variable (Flog Grep Hak Jinx : Set U)

-- Define the given conditions
variable (h1 : Flog ⊆ Grep)
variable (h2 : Hak ⊆ Grep)
variable (h3 : Hak ⊆ Jinx)
variable (h4 : Flog ∩ Jinx = ∅)

-- Theorem to prove
theorem jinx_hak_not_flog : 
  Jinx ⊆ Hak ∧ ∃ x, x ∈ Jinx ∧ x ∉ Flog :=
sorry

end jinx_hak_not_flog_l1400_140022


namespace quadratic_solution_l1400_140030

theorem quadratic_solution (b : ℝ) : 
  ((-10 : ℝ)^2 + b * (-10) - 30 = 0) → b = 7 := by
  sorry

end quadratic_solution_l1400_140030


namespace high_school_total_students_l1400_140031

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  freshman_sample : ℕ
  sophomore_sample : ℕ

/-- The conditions of the problem -/
def problem_conditions : HighSchool where
  senior_students := 600
  sample_size := 90
  freshman_sample := 27
  sophomore_sample := 33
  total_students := 1800  -- This is what we want to prove

theorem high_school_total_students :
  ∀ (hs : HighSchool),
  hs.senior_students = 600 →
  hs.sample_size = 90 →
  hs.freshman_sample = 27 →
  hs.sophomore_sample = 33 →
  hs.total_students = 1800 :=
by
  sorry

#check high_school_total_students

end high_school_total_students_l1400_140031


namespace imaginary_part_of_z_l1400_140002

theorem imaginary_part_of_z (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (Complex.mk 1 a) = Real.sqrt 5) :
  a = 2 := by
sorry

end imaginary_part_of_z_l1400_140002


namespace fraction_equality_implication_l1400_140054

theorem fraction_equality_implication (a b c d : ℝ) :
  (a + b) / (b + c) = (c + d) / (d + a) →
  (a = c ∨ a + b + c + d = 0) :=
by sorry

end fraction_equality_implication_l1400_140054


namespace min_value_cos_sum_l1400_140059

theorem min_value_cos_sum (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ π/2) 
  (hy : 0 ≤ y ∧ y ≤ π/2) (hz : 0 ≤ z ∧ z ≤ π/2) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ 1 := by
  sorry

end min_value_cos_sum_l1400_140059


namespace cosine_difference_l1400_140040

theorem cosine_difference (α β : Real) 
  (h1 : Real.sin α - Real.sin β = 1/2) 
  (h2 : Real.cos α - Real.cos β = 1/3) : 
  Real.cos (α - β) = 59/72 := by
  sorry

end cosine_difference_l1400_140040


namespace two_fifths_of_number_l1400_140021

theorem two_fifths_of_number (x : ℚ) : (2 / 9 : ℚ) * x = 10 → (2 / 5 : ℚ) * x = 18 := by
  sorry

end two_fifths_of_number_l1400_140021


namespace purely_imaginary_z_implies_tan_theta_minus_pi_4_l1400_140057

theorem purely_imaginary_z_implies_tan_theta_minus_pi_4 (θ : ℝ) :
  let z : ℂ := (Real.cos θ - 4/5) + (Real.sin θ - 3/5) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan (θ - π/4) = -7 := by
  sorry

end purely_imaginary_z_implies_tan_theta_minus_pi_4_l1400_140057


namespace unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2_l1400_140084

theorem unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 18 * k) ∧ 
    (28 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 28.2) :=
by
  -- The proof would go here
  sorry

end unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2_l1400_140084


namespace mean_median_difference_l1400_140096

/-- Represents the score distribution of students in a test --/
structure ScoreDistribution where
  score65 : Float
  score75 : Float
  score88 : Float
  score92 : Float
  score100 : Float
  total_percentage : Float
  h_total : total_percentage = score65 + score75 + score88 + score92 + score100

/-- Calculates the median score given a ScoreDistribution --/
def median (sd : ScoreDistribution) : Float :=
  sorry

/-- Calculates the mean score given a ScoreDistribution --/
def mean (sd : ScoreDistribution) : Float :=
  sorry

/-- The main theorem stating the difference between mean and median --/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.score65 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score88 = 0.25)
  (h4 : sd.score92 = 0.10)
  (h5 : sd.score100 = 0.30)
  (h6 : sd.total_percentage = 1.0) :
  mean sd - median sd = -2 :=
sorry

end mean_median_difference_l1400_140096


namespace square_5_on_top_l1400_140020

/-- Represents a square on the paper grid -/
structure Square :=
  (number : Nat)
  (row : Nat)
  (col : Nat)

/-- Represents the paper grid -/
def Grid := List Square

/-- Defines the initial configuration of the grid -/
def initialGrid : Grid :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20].map
    (fun n => ⟨n, (n - 1) / 5 + 1, (n - 1) % 5 + 1⟩)

/-- Performs a folding operation on the grid -/
def fold (g : Grid) (foldType : String) : Grid := sorry

/-- Theorem stating that after all folding operations, square 5 is on top -/
theorem square_5_on_top (g : Grid) (h : g = initialGrid) :
  (fold (fold (fold (fold g "left_third") "right_third") "bottom_half") "top_half").head?.map Square.number = some 5 := by sorry

end square_5_on_top_l1400_140020


namespace solve_equation_l1400_140015

theorem solve_equation (y : ℝ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 := by
  sorry

end solve_equation_l1400_140015


namespace triangle_inequality_sum_negative_l1400_140069

theorem triangle_inequality_sum_negative 
  (a b c x y z : ℝ) 
  (h1 : 0 < b - c) 
  (h2 : b - c < a) 
  (h3 : a < b + c) 
  (h4 : a * x + b * y + c * z = 0) : 
  a * y * z + b * z * x + c * x * y < 0 := by
  sorry

end triangle_inequality_sum_negative_l1400_140069


namespace least_product_of_primes_above_30_l1400_140008

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 30 ∧ q > 30 ∧
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ p' q' : ℕ,
      p'.Prime → q'.Prime →
      p' > 30 → q' > 30 →
      p' ≠ q' →
      p' * q' ≥ 1147 :=
by sorry

end least_product_of_primes_above_30_l1400_140008


namespace equation_solution_l1400_140060

theorem equation_solution : ∃ x : ℝ, (0.82 : ℝ)^3 - (0.1 : ℝ)^3 / (0.82 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.72 := by
  sorry

end equation_solution_l1400_140060


namespace exists_x0_implies_a_value_l1400_140007

noncomputable def f (a x : ℝ) : ℝ := Real.exp (x + a) + x

noncomputable def g (a x : ℝ) : ℝ := Real.log (x + 3) - 4 * Real.exp (-x - a)

theorem exists_x0_implies_a_value (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 2) → a = 2 + Real.log 2 := by
  sorry

end exists_x0_implies_a_value_l1400_140007


namespace train_length_train_length_proof_l1400_140039

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train traveling at 60 km/hr and crossing a pole in 18 seconds has a length of approximately 300.06 meters -/
theorem train_length_proof (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |train_length 60 18 - 300.06| < δ :=
sorry

end train_length_train_length_proof_l1400_140039


namespace sisters_phone_sale_total_l1400_140058

def phone_price : ℕ := 400

theorem sisters_phone_sale_total (vivienne_phones aliyah_extra_phones : ℕ) :
  vivienne_phones = 40 →
  aliyah_extra_phones = 10 →
  (vivienne_phones + (vivienne_phones + aliyah_extra_phones)) * phone_price = 36000 :=
by sorry

end sisters_phone_sale_total_l1400_140058


namespace awards_distribution_count_l1400_140092

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution_count :
  distribute_awards 6 4 = 3720 :=
by sorry

end awards_distribution_count_l1400_140092


namespace candy_distribution_l1400_140035

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 15 →
  num_bags = 5 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 3 := by
  sorry

end candy_distribution_l1400_140035


namespace quadratic_equations_unique_solution_l1400_140033

/-- Given two quadratic equations and their solution sets, prove the coefficients -/
theorem quadratic_equations_unique_solution :
  ∀ (p q r : ℝ),
  let A := {x : ℝ | x^2 + p*x - 2 = 0}
  let B := {x : ℝ | x^2 + q*x + r = 0}
  (A ∪ B = {-2, 1, 5}) →
  (A ∩ B = {-2}) →
  (p = -1 ∧ q = -3 ∧ r = -10) :=
by sorry

end quadratic_equations_unique_solution_l1400_140033


namespace alpha_plus_beta_eq_128_l1400_140027

theorem alpha_plus_beta_eq_128 :
  ∀ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2209) / (x^2 + 66*x - 3969)) →
  α + β = 128 := by
sorry

end alpha_plus_beta_eq_128_l1400_140027


namespace expression_simplification_l1400_140081

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x - 2)) = 1 / 2 := by
  sorry

end expression_simplification_l1400_140081


namespace approval_ratio_rounded_l1400_140048

/-- The ratio of regions needed for approval to total regions -/
def approval_ratio : ℚ := 8 / 15

/-- Rounding a rational number to the nearest tenth -/
def round_to_tenth (q : ℚ) : ℚ := 
  ⌊q * 10 + 1/2⌋ / 10

theorem approval_ratio_rounded : round_to_tenth approval_ratio = 1/2 := by
  sorry

end approval_ratio_rounded_l1400_140048


namespace chord_length_l1400_140038

-- Define the circle C
def Circle (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - m)^2 + (p.2 - n)^2 = 4}

-- Define the theorem
theorem chord_length
  (m n : ℝ) -- Center of the circle
  (A B : ℝ × ℝ) -- Points on the circle
  (hA : A ∈ Circle m n) -- A is on the circle
  (hB : B ∈ Circle m n) -- B is on the circle
  (hAB : A ≠ B) -- A and B are different points
  (h_sum : ‖(A.1 - m, A.2 - n) + (B.1 - m, B.2 - n)‖ = 2 * Real.sqrt 3) -- |→CA + →CB| = 2√3
  : ‖(A.1 - B.1, A.2 - B.2)‖ = 2 := by
  sorry

end chord_length_l1400_140038


namespace intersection_A_complement_B_l1400_140026

-- Define the sets A and B
def A : Set ℝ := {x | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | -Real.sqrt 2 ≤ x ∧ x ≤ -1} := by sorry

end intersection_A_complement_B_l1400_140026


namespace tom_dance_lessons_l1400_140014

theorem tom_dance_lessons 
  (cost_per_lesson : ℕ) 
  (free_lessons : ℕ) 
  (total_paid : ℕ) :
  cost_per_lesson = 10 →
  free_lessons = 2 →
  total_paid = 80 →
  (total_paid / cost_per_lesson) + free_lessons = 10 :=
by sorry

end tom_dance_lessons_l1400_140014


namespace problem_statement_l1400_140064

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x > 0, x > 0 → 6 - 1 / x ≤ 9 * x) ∧
  (a^2 + 9 * b^2 + 2 * a * b = a^2 * b^2 → a * b ≥ 8) := by
  sorry

end problem_statement_l1400_140064


namespace composite_n_fourth_plus_64_l1400_140003

theorem composite_n_fourth_plus_64 : ∃ (n : ℕ), ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64 = a * b :=
sorry

end composite_n_fourth_plus_64_l1400_140003


namespace coin_flip_expected_value_is_two_thirds_l1400_140082

def coin_flip_expected_value : ℚ :=
  let p_heads : ℚ := 1/2
  let p_tails : ℚ := 1/3
  let p_edge : ℚ := 1/6
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let win_edge : ℚ := -5
  p_heads * win_heads + p_tails * win_tails + p_edge * win_edge

theorem coin_flip_expected_value_is_two_thirds :
  coin_flip_expected_value = 2/3 := by
  sorry

end coin_flip_expected_value_is_two_thirds_l1400_140082


namespace total_purchase_ways_l1400_140056

def oreo_flavors : ℕ := 7
def milk_types : ℕ := 4
def total_items : ℕ := 5

def ways_to_purchase : ℕ := sorry

theorem total_purchase_ways :
  ways_to_purchase = 13279 := by sorry

end total_purchase_ways_l1400_140056


namespace expression_equality_l1400_140073

theorem expression_equality (y θ Q : ℝ) (h : 5 * (3 * y + 7 * Real.sin θ) = Q) :
  15 * (9 * y + 21 * Real.sin θ) = 9 * Q := by
  sorry

end expression_equality_l1400_140073


namespace circle_angle_sum_l1400_140029

theorem circle_angle_sum (a b : ℝ) : 
  a + b + 110 + 60 = 360 → a + b = 190 := by
  sorry

end circle_angle_sum_l1400_140029


namespace ratio_from_percentage_l1400_140071

theorem ratio_from_percentage (x y : ℝ) (h : y = x * (1 - 0.909090909090909)) :
  x / y = 11 := by
sorry

end ratio_from_percentage_l1400_140071


namespace common_external_tangent_y_intercept_l1400_140083

/-- Given two circles with centers (1,3) and (15,8) and radii 3 and 10 respectively,
    this theorem proves that the y-intercept of their common external tangent
    with positive slope is 518/1197. -/
theorem common_external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (1, 3)
  let r1 : ℝ := 3
  let c2 : ℝ × ℝ := (15, 8)
  let r2 : ℝ := 10
  let m : ℝ := (8 - 3) / (15 - 1)  -- slope of line connecting centers
  let tan_2theta : ℝ := (2 * m) / (1 - m^2)  -- tangent of double angle
  let m_tangent : ℝ := Real.sqrt (tan_2theta / (1 + tan_2theta))  -- slope of tangent line
  let x_intercept : ℝ := -(3 - m * 1) / m  -- x-intercept of line connecting centers
  ∃ b : ℝ, b = m_tangent * (-x_intercept) ∧ b = 518 / 1197 :=
by sorry

end common_external_tangent_y_intercept_l1400_140083


namespace no_equal_division_of_scalene_triangle_l1400_140075

/-- A triangle represented by its three vertices in ℝ² -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- A triangle is scalene if all its sides have different lengths -/
def isScalene (t : Triangle) : Prop := sorry

/-- A point D that divides the triangle into two equal parts -/
def dividingPoint (t : Triangle) (D : ℝ × ℝ) : Prop :=
  triangleArea ⟨t.A, t.B, D⟩ = triangleArea ⟨t.A, t.C, D⟩

/-- Theorem: A scalene triangle cannot be divided into two equal triangles -/
theorem no_equal_division_of_scalene_triangle (t : Triangle) :
  isScalene t → ¬∃ D : ℝ × ℝ, dividingPoint t D := by
  sorry

end no_equal_division_of_scalene_triangle_l1400_140075


namespace inverse_variation_problem_l1400_140010

theorem inverse_variation_problem (y z : ℝ) (h1 : y^4 * z^(1/4) = 162) (h2 : y = 6) : z = 1/4096 := by
  sorry

end inverse_variation_problem_l1400_140010


namespace complement_intersection_theorem_l1400_140085

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {0, 2, 5}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0, 5} := by
  sorry

end complement_intersection_theorem_l1400_140085


namespace min_reciprocal_sum_l1400_140095

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  1/x + 1/y + 1/z ≥ 9/2 := by
sorry

end min_reciprocal_sum_l1400_140095


namespace sector_perimeter_l1400_140011

/-- Given a circular sector with area 2 cm² and central angle 4 radians, its perimeter is 6 cm. -/
theorem sector_perimeter (A : ℝ) (α : ℝ) (P : ℝ) : 
  A = 2 → α = 4 → P = 2 * Real.sqrt (2 / α) + Real.sqrt (2 / α) * α → P = 6 := by
  sorry

end sector_perimeter_l1400_140011


namespace max_area_inscribed_ngon_l1400_140074

/-- An n-gon with given side lengths -/
structure Ngon (n : ℕ) where
  sides : Fin n → ℝ
  area : ℝ

/-- An n-gon inscribed in a circle -/
structure InscribedNgon (n : ℕ) extends Ngon n where
  isInscribed : Bool

/-- Theorem: The area of any n-gon is less than or equal to 
    the area of the inscribed n-gon with the same side lengths -/
theorem max_area_inscribed_ngon (n : ℕ) (l : Fin n → ℝ) :
  ∀ (P : Ngon n), P.sides = l →
  ∃ (Q : InscribedNgon n), Q.sides = l ∧ P.area ≤ Q.area :=
sorry

end max_area_inscribed_ngon_l1400_140074


namespace no_solution_when_k_is_seven_l1400_140047

theorem no_solution_when_k_is_seven (k : ℝ) (h : k = 7) :
  ¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 5 ∧ (x^2 - 1) / (x - 3) = (x^2 - k) / (x - 5) :=
by
  sorry

end no_solution_when_k_is_seven_l1400_140047


namespace quadratic_solution_difference_squared_l1400_140051

theorem quadratic_solution_difference_squared : 
  ∀ Φ φ : ℝ, 
  Φ ≠ φ → 
  Φ^2 - 3*Φ + 1 = 0 → 
  φ^2 - 3*φ + 1 = 0 → 
  (Φ - φ)^2 = 5 := by
sorry

end quadratic_solution_difference_squared_l1400_140051


namespace quadratic_factorization_l1400_140070

theorem quadratic_factorization (x : ℝ) : x^2 - 30*x + 225 = (x - 15)^2 := by
  sorry

end quadratic_factorization_l1400_140070


namespace contest_sequences_equal_combination_l1400_140088

/-- Represents the number of players in each team -/
def team_size : ℕ := 7

/-- Represents the total number of players from both teams -/
def total_players : ℕ := 2 * team_size

/-- Represents the number of different possible sequences of matches in the contest -/
def match_sequences : ℕ := Nat.choose total_players team_size

theorem contest_sequences_equal_combination :
  match_sequences = 3432 := by
  sorry

end contest_sequences_equal_combination_l1400_140088


namespace tony_bread_slices_left_l1400_140078

/-- The number of slices of bread Tony uses in a week -/
def bread_used (weekday_slices : ℕ) (saturday_slices : ℕ) (sunday_slices : ℕ) : ℕ :=
  5 * weekday_slices + saturday_slices + sunday_slices

/-- The number of slices left from a loaf -/
def slices_left (total_slices : ℕ) (used_slices : ℕ) : ℕ :=
  total_slices - used_slices

/-- Theorem stating the number of slices left from Tony's bread usage -/
theorem tony_bread_slices_left :
  let weekday_slices := 2
  let saturday_slices := 5
  let sunday_slices := 1
  let total_slices := 22
  let used_slices := bread_used weekday_slices saturday_slices sunday_slices
  slices_left total_slices used_slices = 6 := by
  sorry

end tony_bread_slices_left_l1400_140078


namespace quadratic_trinomial_existence_l1400_140009

theorem quadratic_trinomial_existence (a b c : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (p q r : ℤ), p > 0 ∧ 
    (∀ x : ℤ, p * x^2 + q * x + r = x^3 - (x - a) * (x - b) * (x - c)) ∧
    (p * a^2 + q * a + r = a^3) ∧
    (p * b^2 + q * b + r = b^3) ∧
    (p * c^2 + q * c + r = c^3) := by
  sorry

end quadratic_trinomial_existence_l1400_140009


namespace sqrt_division_equality_l1400_140028

theorem sqrt_division_equality : Real.sqrt 3 / Real.sqrt 5 = Real.sqrt 15 / 5 := by
  sorry

end sqrt_division_equality_l1400_140028


namespace length_AB_squared_l1400_140053

/-- The parabola function y = 3x^2 + 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 2

/-- The square of the distance between two points -/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x2 - x1)^2 + (y2 - y1)^2

theorem length_AB_squared :
  ∀ (x1 y1 x2 y2 : ℝ),
  f x1 = y1 →  -- Point A is on the parabola
  f x2 = y2 →  -- Point B is on the parabola
  (x1 + x2) / 2 = 1 →  -- x-coordinate of midpoint C
  (y1 + y2) / 2 = 1 →  -- y-coordinate of midpoint C
  distance_squared x1 y1 x2 y2 = 17 := by
    sorry

end length_AB_squared_l1400_140053


namespace overlap_area_theorem_l1400_140086

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side : ℝ)
  (rotation : ℝ)

/-- Calculates the area of overlap between three rotated square sheets -/
def area_of_overlap (s1 s2 s3 : Sheet) : ℝ :=
  sorry

/-- The theorem stating the area of overlap for the given problem -/
theorem overlap_area_theorem :
  let s1 : Sheet := { side := 8, rotation := 0 }
  let s2 : Sheet := { side := 8, rotation := 45 }
  let s3 : Sheet := { side := 8, rotation := 90 }
  area_of_overlap s1 s2 s3 = 96 :=
sorry

end overlap_area_theorem_l1400_140086


namespace alpha_value_l1400_140025

/-- Given that α is an acute angle and sin(α - 10°) = √3/2, prove that α = 70°. -/
theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < 90) (h2 : Real.sin (α - 10) = Real.sqrt 3 / 2) : 
  α = 70 := by
  sorry

end alpha_value_l1400_140025


namespace smallest_marble_collection_l1400_140050

theorem smallest_marble_collection (M : ℕ) : 
  M > 1 → 
  M % 5 = 2 → 
  M % 6 = 2 → 
  M % 7 = 2 → 
  (∀ n : ℕ, n > 1 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ n % 7 = 2 → n ≥ M) → 
  M = 212 :=
by sorry

end smallest_marble_collection_l1400_140050


namespace p_necessary_not_sufficient_l1400_140062

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- Define the condition p
def condition_p (m : ℝ) : Prop := -1 < m ∧ m < 5

-- Define the condition q
def condition_q (m : ℝ) : Prop :=
  ∀ x, quadratic_equation m x = 0 → -2 < x ∧ x < 4

-- Theorem: p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ m, condition_q m → condition_p m) ∧
  ¬(∀ m, condition_p m → condition_q m) :=
sorry

end p_necessary_not_sufficient_l1400_140062


namespace simplify_expression_l1400_140012

theorem simplify_expression (c : ℝ) : ((3 * c + 6) - 6 * c) / 3 = -c + 2 := by
  sorry

end simplify_expression_l1400_140012


namespace toby_monday_steps_l1400_140055

/-- Represents the number of steps walked on each day of the week -/
structure WeekSteps where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total steps walked in a week -/
def totalSteps (w : WeekSteps) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

/-- Calculates the average steps per day in a week -/
def averageSteps (w : WeekSteps) : ℚ :=
  (totalSteps w : ℚ) / 7

theorem toby_monday_steps (w : WeekSteps) 
  (h1 : averageSteps w = 9000)
  (h2 : w.sunday = 9400)
  (h3 : w.tuesday = 8300)
  (h4 : w.wednesday = 9200)
  (h5 : w.thursday = 8900)
  (h6 : (w.friday + w.saturday : ℚ) / 2 = 9050) :
  w.monday = 9100 := by
  sorry


end toby_monday_steps_l1400_140055


namespace temperature_conversion_l1400_140065

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 122 → t = 50 := by
  sorry

end temperature_conversion_l1400_140065


namespace stratified_sampling_size_l1400_140077

def workshop_A : ℕ := 120
def workshop_B : ℕ := 80
def workshop_C : ℕ := 60

def total_production : ℕ := workshop_A + workshop_B + workshop_C

def sample_size_C : ℕ := 3

theorem stratified_sampling_size :
  (workshop_C : ℚ) / total_production = sample_size_C / (13 : ℚ) := by sorry

end stratified_sampling_size_l1400_140077


namespace extended_pattern_black_tiles_l1400_140080

/-- Represents a square pattern of tiles -/
structure SquarePattern :=
  (size : Nat)
  (blackTiles : Nat)

/-- Extends a square pattern by adding a border of black tiles -/
def extendPattern (pattern : SquarePattern) : SquarePattern :=
  { size := pattern.size + 2,
    blackTiles := pattern.blackTiles + (pattern.size + 2) * 4 - 4 }

theorem extended_pattern_black_tiles :
  let originalPattern : SquarePattern := { size := 5, blackTiles := 10 }
  let extendedPattern := extendPattern originalPattern
  extendedPattern.blackTiles = 34 := by
  sorry

end extended_pattern_black_tiles_l1400_140080


namespace ellipse_perpendicular_bisector_bound_l1400_140036

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the theorem
theorem ellipse_perpendicular_bisector_bound 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (A B : ℝ × ℝ) 
  (h_A : is_on_ellipse A.1 A.2 a b) 
  (h_B : is_on_ellipse B.1 B.2 a b) 
  (x₀ : ℝ) 
  (h_perp_bisector : ∃ (k : ℝ), 
    k * (A.1 - B.1) = A.2 - B.2 ∧ 
    x₀ = (A.1 + B.1) / 2 + k * (A.2 + B.2) / 2) :
  -((a^2 - b^2) / a) < x₀ ∧ x₀ < (a^2 - b^2) / a :=
sorry

end ellipse_perpendicular_bisector_bound_l1400_140036


namespace exam_maximum_marks_l1400_140004

theorem exam_maximum_marks :
  let pass_percentage : ℚ := 45 / 100
  let fail_score : ℕ := 180
  let fail_margin : ℕ := 45
  let max_marks : ℕ := 500
  (pass_percentage * max_marks = fail_score + fail_margin) ∧
  (pass_percentage * max_marks = (fail_score + fail_margin : ℚ)) :=
by sorry

end exam_maximum_marks_l1400_140004


namespace complex_magnitude_problem_l1400_140066

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l1400_140066


namespace divisor_power_equation_l1400_140023

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- The statement of the problem -/
theorem divisor_power_equation :
  ∀ n k : ℕ+, ∀ p : ℕ,
  Prime p →
  (n : ℕ) ^ (d n) - 1 = p ^ (k : ℕ) →
  ((n = 2 ∧ k = 1 ∧ p = 3) ∨ (n = 3 ∧ k = 3 ∧ p = 2)) :=
by sorry

end divisor_power_equation_l1400_140023


namespace complement_implies_set_l1400_140087

def U : Set ℕ := {1, 3, 5, 7}

theorem complement_implies_set (M : Set ℕ) : 
  U = {1, 3, 5, 7} → (U \ M = {5, 7}) → M = {1, 3} := by
  sorry

end complement_implies_set_l1400_140087


namespace geometric_sequence_problem_l1400_140052

theorem geometric_sequence_problem (a b c : ℝ) :
  (∀ q : ℝ, 1 * q = a ∧ a * q = b ∧ b * q = c ∧ c * q = 4) → b = 2 := by
  sorry

end geometric_sequence_problem_l1400_140052


namespace compute_fraction_power_and_multiply_l1400_140001

theorem compute_fraction_power_and_multiply :
  8 * (1 / 7)^2 = 8 / 49 := by
  sorry

end compute_fraction_power_and_multiply_l1400_140001


namespace division_reduction_l1400_140043

theorem division_reduction (original : ℕ) (divisor : ℕ) (reduction : ℕ) : 
  original = 72 → divisor = 3 → reduction = 48 → 
  (original : ℚ) / divisor = original - reduction :=
by
  sorry

end division_reduction_l1400_140043


namespace mans_speed_with_current_is_15_l1400_140016

/-- 
Given a current speed and a man's speed against the current,
calculate the man's speed with the current.
-/
def mans_speed_with_current (current_speed : ℝ) (speed_against_current : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- 
Theorem: Given a current speed of 2.5 km/hr and a speed against the current of 10 km/hr,
the man's speed with the current is 15 km/hr.
-/
theorem mans_speed_with_current_is_15 :
  mans_speed_with_current 2.5 10 = 15 := by
  sorry

#eval mans_speed_with_current 2.5 10

end mans_speed_with_current_is_15_l1400_140016


namespace inequality_proof_l1400_140037

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  2 / ((a + b) * (c + d)) ≤ 1 / Real.sqrt (a * b) + 1 / Real.sqrt (c * d) := by
  sorry

end inequality_proof_l1400_140037


namespace perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_perpendicular_parallel_l1400_140045

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_lines_from_perpendicular_planes
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : perpendicular_line_plane n β)
  (h3 : perpendicular_plane_plane α β) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem perpendicular_lines_from_perpendicular_parallel
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_line_plane n β)
  (h3 : parallel_plane_plane α β) :
  perpendicular_line_line m n :=
sorry

end perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_perpendicular_parallel_l1400_140045


namespace power_series_identity_l1400_140079

/-- Given that (1 - hx)⁻¹ (1 - kx)⁻¹ = ∑(i≥0) aᵢ xⁱ, 
    prove that (1 + hkx)(1 - hkx)⁻¹ (1 - h²x)⁻¹ (1 - k²x)⁻¹ = ∑(i≥0) aᵢ² xⁱ -/
theorem power_series_identity 
  (h k : ℝ) (x : ℝ) (a : ℕ → ℝ) :
  (∀ x, (1 - h*x)⁻¹ * (1 - k*x)⁻¹ = ∑' i, a i * x^i) →
  (1 + h*k*x) * (1 - h*k*x)⁻¹ * (1 - h^2*x)⁻¹ * (1 - k^2*x)⁻¹ = ∑' i, (a i)^2 * x^i :=
by
  sorry

end power_series_identity_l1400_140079


namespace function_identity_l1400_140018

theorem function_identity (f : ℕ → ℕ) :
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) ≤ x * (1 + f y)) →
  ∀ x : ℕ, f x = x :=
by sorry

end function_identity_l1400_140018
