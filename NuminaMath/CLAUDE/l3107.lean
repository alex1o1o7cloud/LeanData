import Mathlib

namespace sum_of_factors_of_30_l3107_310705

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by
  sorry

end sum_of_factors_of_30_l3107_310705


namespace tan_neg_five_pi_fourth_l3107_310709

theorem tan_neg_five_pi_fourth : Real.tan (-5 * π / 4) = -1 := by
  sorry

end tan_neg_five_pi_fourth_l3107_310709


namespace james_earnings_l3107_310729

/-- James' earnings problem -/
theorem james_earnings (january : ℕ) (february : ℕ) (march : ℕ) 
  (h1 : february = 2 * january)
  (h2 : march = february - 2000)
  (h3 : january + february + march = 18000) :
  january = 4000 := by
  sorry

end james_earnings_l3107_310729


namespace sin_480_plus_tan_300_l3107_310757

/-- The sum of sine of 480 degrees and tangent of 300 degrees equals negative square root of 3 divided by 2. -/
theorem sin_480_plus_tan_300 : Real.sin (480 * π / 180) + Real.tan (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_480_plus_tan_300_l3107_310757


namespace T_5_value_l3107_310766

/-- An arithmetic sequence with first term 1 and common difference 1 -/
def a (n : ℕ) : ℚ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ := n * (n + 1) / 2

/-- Sum of the first n terms of the sequence {1/S_n} -/
def T (n : ℕ) : ℚ := 2 * n / (n + 1)

/-- Theorem: T_5 = 5/3 -/
theorem T_5_value : T 5 = 5 / 3 := by sorry

end T_5_value_l3107_310766


namespace quadratic_inequality_solution_l3107_310731

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    if its solution set for f(x) > 0 is (-1, 2), 
    then b + c = -3 -/
theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x, x^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) → 
  b + c = -3 := by sorry

end quadratic_inequality_solution_l3107_310731


namespace prime_pythagorean_inequality_l3107_310798

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (hp : Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
sorry

end prime_pythagorean_inequality_l3107_310798


namespace sand_pile_volume_l3107_310763

/-- The volume of a cone with diameter 12 feet and height 60% of the diameter is 86.4π cubic feet -/
theorem sand_pile_volume : 
  let diameter : ℝ := 12
  let height : ℝ := 0.6 * diameter
  let radius : ℝ := diameter / 2
  let volume : ℝ := (1/3) * π * radius^2 * height
  volume = 86.4 * π := by sorry

end sand_pile_volume_l3107_310763


namespace total_cost_of_tickets_l3107_310787

def total_tickets : ℕ := 29
def cheap_ticket_price : ℕ := 7
def expensive_ticket_price : ℕ := 9
def expensive_tickets : ℕ := 11

theorem total_cost_of_tickets : 
  cheap_ticket_price * (total_tickets - expensive_tickets) + 
  expensive_ticket_price * expensive_tickets = 225 := by sorry

end total_cost_of_tickets_l3107_310787


namespace miranda_pillow_stuffing_l3107_310730

/-- 
Given:
- Two pounds of feathers are needed for each pillow
- A pound of goose feathers is approximately 300 feathers
- A pound of duck feathers is approximately 500 feathers
- Miranda's goose has approximately 3600 feathers
- Miranda's duck has approximately 4000 feathers

Prove that Miranda can stuff 10 pillows.
-/
theorem miranda_pillow_stuffing (
  feathers_per_pillow : ℕ)
  (goose_feathers_per_pound : ℕ)
  (duck_feathers_per_pound : ℕ)
  (goose_total_feathers : ℕ)
  (duck_total_feathers : ℕ)
  (h1 : feathers_per_pillow = 2)
  (h2 : goose_feathers_per_pound = 300)
  (h3 : duck_feathers_per_pound = 500)
  (h4 : goose_total_feathers = 3600)
  (h5 : duck_total_feathers = 4000) :
  (goose_total_feathers / goose_feathers_per_pound + 
   duck_total_feathers / duck_feathers_per_pound) / 
  feathers_per_pillow = 10 := by
  sorry

end miranda_pillow_stuffing_l3107_310730


namespace remainder_17_pow_53_mod_7_l3107_310793

theorem remainder_17_pow_53_mod_7 : 17^53 % 7 = 5 := by
  sorry

end remainder_17_pow_53_mod_7_l3107_310793


namespace cube_side_ratio_l3107_310783

theorem cube_side_ratio (s S : ℝ) (h : s > 0) (H : S > 0) :
  (6 * S^2) / (6 * s^2) = 9 → S / s = 3 := by
sorry

end cube_side_ratio_l3107_310783


namespace pet_beds_per_pet_l3107_310773

theorem pet_beds_per_pet (total_beds : ℕ) (num_pets : ℕ) (beds_per_pet : ℕ) : 
  total_beds = 20 → num_pets = 10 → beds_per_pet = total_beds / num_pets → beds_per_pet = 2 := by
  sorry

end pet_beds_per_pet_l3107_310773


namespace courtyard_paving_l3107_310718

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℕ := 2500
def courtyard_width : ℕ := 1800

-- Define the brick dimensions in centimeters
def brick_length : ℕ := 20
def brick_width : ℕ := 10

-- Define the function to calculate the number of bricks required
def bricks_required (cl cw bl bw : ℕ) : ℕ :=
  (cl * cw) / (bl * bw)

-- Theorem statement
theorem courtyard_paving :
  bricks_required courtyard_length courtyard_width brick_length brick_width = 22500 := by
  sorry

end courtyard_paving_l3107_310718


namespace arithmetic_sequence_term_count_l3107_310737

/-- 
Given an arithmetic sequence with:
- First term a = 5
- Last term l = 203
- Common difference d = 3

Prove that the number of terms in this sequence is 67.
-/
theorem arithmetic_sequence_term_count : 
  ∀ (a l d n : ℕ), 
  a = 5 → l = 203 → d = 3 → 
  l = a + (n - 1) * d → 
  n = 67 := by
sorry

end arithmetic_sequence_term_count_l3107_310737


namespace count_fractions_is_36_l3107_310708

/-- A function that counts the number of fractions less than 1 with single-digit numerators and denominators -/
def count_fractions : ℕ := 
  let single_digit (n : ℕ) := n ≥ 1 ∧ n ≤ 9
  let is_valid_fraction (n d : ℕ) := single_digit n ∧ single_digit d ∧ n < d
  (Finset.sum (Finset.range 9) (λ d => 
    (Finset.filter (λ n => is_valid_fraction n (d + 1)) (Finset.range (d + 1))).card
  ))

/-- Theorem stating that the count of fractions less than 1 with single-digit numerators and denominators is 36 -/
theorem count_fractions_is_36 : count_fractions = 36 := by
  sorry

end count_fractions_is_36_l3107_310708


namespace folded_paper_distance_l3107_310745

/-- Given a square sheet of paper with area 18 cm², prove that when folded such that
    the visible black area equals the visible white area, the distance from the folded
    point to its original position is 2√6 cm. -/
theorem folded_paper_distance (side_length : ℝ) (fold_length : ℝ) (distance : ℝ) : 
  side_length^2 = 18 →
  fold_length^2 = 12 →
  (1/2) * fold_length^2 = 18 - fold_length^2 →
  distance^2 = 2 * fold_length^2 →
  distance = 2 * Real.sqrt 6 :=
by sorry

end folded_paper_distance_l3107_310745


namespace tangent_line_at_x_1_l3107_310795

noncomputable def f (x : ℝ) : ℝ := (x^5 + 1) / (x^4 + 1)

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := deriv f x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ → y = (1/2) * x + 1/2 :=
by sorry

end tangent_line_at_x_1_l3107_310795


namespace inequality_proof_l3107_310756

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : 1/x + 1/y + 1/z = 1) : 
  Real.sqrt (x + y*z) + Real.sqrt (y + z*x) + Real.sqrt (z + x*y) ≥ 
  Real.sqrt (x*y*z) + Real.sqrt x + Real.sqrt y + Real.sqrt z := by
sorry

end inequality_proof_l3107_310756


namespace equal_area_centroid_l3107_310733

/-- Given a triangle PQR with vertices P(4,3), Q(-1,6), and R(7,-2),
    if point S(x,y) is chosen such that triangles PQS, PRS, and QRS have equal areas,
    then 8x + 3y = 101/3 -/
theorem equal_area_centroid (x y : ℚ) : 
  let P : ℚ × ℚ := (4, 3)
  let Q : ℚ × ℚ := (-1, 6)
  let R : ℚ × ℚ := (7, -2)
  let S : ℚ × ℚ := (x, y)
  let area (A B C : ℚ × ℚ) : ℚ := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area P Q S = area P R S ∧ area P R S = area Q R S →
  8 * x + 3 * y = 101 / 3 :=
by sorry

end equal_area_centroid_l3107_310733


namespace tourist_guide_groupings_l3107_310765

/-- The number of ways to distribute n distinguishable objects into 2 non-empty groups -/
def distributionCount (n : ℕ) : ℕ :=
  2^n - 2

/-- The number of tourists -/
def numTourists : ℕ := 6

/-- The number of guides -/
def numGuides : ℕ := 2

theorem tourist_guide_groupings :
  distributionCount numTourists = 62 :=
sorry

end tourist_guide_groupings_l3107_310765


namespace total_cost_after_discounts_l3107_310789

def dozen : ℕ := 12

def red_roses : ℕ := 2 * dozen
def white_roses : ℕ := 1 * dozen
def yellow_roses : ℕ := 2 * dozen

def red_price : ℚ := 6
def white_price : ℚ := 7
def yellow_price : ℚ := 5

def total_roses : ℕ := red_roses + white_roses + yellow_roses

def initial_cost : ℚ := 
  red_roses * red_price + white_roses * white_price + yellow_roses * yellow_price

def first_discount_rate : ℚ := 15 / 100
def second_discount_rate : ℚ := 10 / 100

theorem total_cost_after_discounts :
  total_roses > 30 ∧ total_roses > 50 →
  let cost_after_first_discount := initial_cost * (1 - first_discount_rate)
  let final_cost := cost_after_first_discount * (1 - second_discount_rate)
  final_cost = 266.22 := by sorry

end total_cost_after_discounts_l3107_310789


namespace equation_solution_l3107_310712

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (x - 1) / (x - 2) + 1 / (2 - x) = 3 ↔ x = -2 := by
  sorry

end equation_solution_l3107_310712


namespace cube_root_of_special_sum_l3107_310794

theorem cube_root_of_special_sum (m n : ℚ) 
  (h : m + 2*n + Real.sqrt 2 * (2 - n) = Real.sqrt 2 * (Real.sqrt 2 + 6) + 15) :
  (((m : ℝ).sqrt + n) ^ 100) ^ (1/3 : ℝ) = 1 :=
sorry

end cube_root_of_special_sum_l3107_310794


namespace solution_set_of_inequality_l3107_310716

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Define the solution set
def S : Set ℝ := {2} ∪ {x | x > 6}

-- Theorem statement
theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = S :=
by sorry

end solution_set_of_inequality_l3107_310716


namespace range_of_a_l3107_310759

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ≤ 0 := by
  sorry

end range_of_a_l3107_310759


namespace cde_value_l3107_310790

/-- Represents the digits in the coding system -/
inductive Digit
| A | B | C | D | E | F

/-- Converts a Digit to its corresponding integer value -/
def digit_to_int : Digit → Nat
| Digit.A => 0
| Digit.B => 5
| Digit.C => 0
| Digit.D => 1
| Digit.E => 0
| Digit.F => 5

/-- Represents a number in the coding system -/
structure EncodedNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (ones : Digit)

/-- Converts an EncodedNumber to its base-10 value -/
def to_base_10 (n : EncodedNumber) : Nat :=
  6^2 * (digit_to_int n.hundreds) + 6 * (digit_to_int n.tens) + (digit_to_int n.ones)

/-- States that BCF, BCE, CAA are consecutive integers -/
axiom consecutive_encoding :
  ∃ (n : Nat),
    to_base_10 (EncodedNumber.mk Digit.B Digit.C Digit.F) = n ∧
    to_base_10 (EncodedNumber.mk Digit.B Digit.C Digit.E) = n + 1 ∧
    to_base_10 (EncodedNumber.mk Digit.C Digit.A Digit.A) = n + 2

theorem cde_value :
  to_base_10 (EncodedNumber.mk Digit.C Digit.D Digit.E) = 6 :=
by sorry

end cde_value_l3107_310790


namespace election_win_margin_l3107_310728

theorem election_win_margin 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (winner_votes : ℕ) 
  (h1 : winner_percentage = 62 / 100)
  (h2 : winner_votes = 744)
  (h3 : ↑winner_votes = winner_percentage * ↑total_votes) :
  winner_votes - (total_votes - winner_votes) = 288 :=
by sorry

end election_win_margin_l3107_310728


namespace rectangle_formations_l3107_310785

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 6

theorem rectangle_formations : 
  (Nat.choose horizontal_lines 2) * (Nat.choose vertical_lines 2) = 150 := by
  sorry

end rectangle_formations_l3107_310785


namespace max_sum_with_square_diff_l3107_310744

theorem max_sum_with_square_diff (a b : ℤ) (h : a^2 - b^2 = 144) :
  ∃ (d : ℤ), d = a + b ∧ d ≤ 72 ∧ ∃ (a' b' : ℤ), a'^2 - b'^2 = 144 ∧ a' + b' = 72 := by
  sorry

end max_sum_with_square_diff_l3107_310744


namespace algebraic_expression_value_l3107_310767

/-- Given an algebraic expression mx^2 - 2x + n that equals 2 when x = 2,
    prove that it equals 10 when x = -2 -/
theorem algebraic_expression_value (m n : ℝ) 
  (h : m * 2^2 - 2 * 2 + n = 2) : 
  m * (-2)^2 - 2 * (-2) + n = 10 := by
  sorry

end algebraic_expression_value_l3107_310767


namespace mother_age_proof_l3107_310711

def id_number : ℕ := 6101131197410232923
def current_year : ℕ := 2014

def extract_birth_year (id : ℕ) : ℕ :=
  (id / 10^13) % 10000

def calculate_age (birth_year current_year : ℕ) : ℕ :=
  current_year - birth_year

theorem mother_age_proof :
  calculate_age (extract_birth_year id_number) current_year = 40 := by
  sorry

end mother_age_proof_l3107_310711


namespace billys_songbook_l3107_310735

/-- The number of songs Billy can play -/
def songs_can_play : ℕ := 24

/-- The number of songs Billy still needs to learn -/
def songs_to_learn : ℕ := 28

/-- The total number of songs in Billy's music book -/
def total_songs : ℕ := songs_can_play + songs_to_learn

theorem billys_songbook :
  total_songs = 52 := by sorry

end billys_songbook_l3107_310735


namespace sum_greater_than_four_l3107_310771

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  m : ℕ+
  n : ℕ+
  h_neq : m ≠ n
  S : ℕ+ → ℚ
  h_m : S m = m / n
  h_n : S n = n / m

/-- The sum of the first (m+n) terms is greater than 4 -/
theorem sum_greater_than_four (seq : ArithmeticSequence) : seq.S (seq.m + seq.n) > 4 := by
  sorry

end sum_greater_than_four_l3107_310771


namespace pine_percentage_correct_l3107_310710

/-- Represents the number of trees of each type in the forest -/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- The total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- The actual composition of the forest -/
def forest : ForestComposition := {
  oak := 720,
  pine := 520,
  spruce := 400,
  birch := 2160
}

/-- The percentage of pine trees in the forest -/
def pine_percentage : ℚ := 13 / 100

theorem pine_percentage_correct :
  (forest.oak + forest.pine + forest.spruce + forest.birch = total_trees) ∧
  (forest.spruce = total_trees / 10) ∧
  (forest.oak = forest.spruce + forest.pine) ∧
  (forest.birch = 2160) →
  (forest.pine : ℚ) / total_trees = pine_percentage :=
by sorry

end pine_percentage_correct_l3107_310710


namespace is_quadratic_equation_l3107_310715

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ (x - 1)^2 = 2*(3 - x)^2 ↔ a*x^2 + b*x + c = 0 :=
sorry

end is_quadratic_equation_l3107_310715


namespace remainder_of_sum_l3107_310724

theorem remainder_of_sum (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end remainder_of_sum_l3107_310724


namespace seventeen_sum_of_two_primes_l3107_310723

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem seventeen_sum_of_two_primes :
  ∃! (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 17 :=
sorry

end seventeen_sum_of_two_primes_l3107_310723


namespace celebration_day_l3107_310727

/-- Given a person born on a Friday, their 1200th day of life will fall on a Saturday -/
theorem celebration_day (birth_day : Nat) (birth_weekday : Nat) : 
  birth_weekday = 5 → (birth_day + 1199) % 7 = 6 := by
  sorry

#check celebration_day

end celebration_day_l3107_310727


namespace base_8_to_10_reverse_digits_l3107_310700

theorem base_8_to_10_reverse_digits : ∃ (d e f : ℕ), 
  (0 ≤ d ∧ d ≤ 7) ∧ 
  (0 ≤ e ∧ e ≤ 7) ∧ 
  (0 ≤ f ∧ f ≤ 7) ∧ 
  e = 3 ∧
  (64 * d + 8 * e + f = 100 * f + 10 * e + d) := by
  sorry

end base_8_to_10_reverse_digits_l3107_310700


namespace colored_paper_difference_l3107_310770

/-- 
Given that Minyoung and Hoseok each start with 150 pieces of colored paper,
Minyoung buys 32 more pieces, and Hoseok buys 49 more pieces,
prove that Hoseok ends up with 17 more pieces than Minyoung.
-/
theorem colored_paper_difference : 
  let initial_paper : ℕ := 150
  let minyoung_bought : ℕ := 32
  let hoseok_bought : ℕ := 49
  let minyoung_total := initial_paper + minyoung_bought
  let hoseok_total := initial_paper + hoseok_bought
  hoseok_total - minyoung_total = 17 := by
  sorry

end colored_paper_difference_l3107_310770


namespace intersection_subset_l3107_310784

theorem intersection_subset (A B : Set α) : A ∩ B = B → B ⊆ A := by
  sorry

end intersection_subset_l3107_310784


namespace pauls_crayons_l3107_310741

theorem pauls_crayons (erasers : ℕ) (crayons_difference : ℕ) :
  erasers = 457 →
  crayons_difference = 66 →
  erasers + crayons_difference = 523 :=
by sorry

end pauls_crayons_l3107_310741


namespace no_function_satisfies_conditions_l3107_310772

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧ 
  (∀ (x : ℝ), x ≠ 0 → f (x + 1/x^2) = f x + (f (1/x))^2) := by
  sorry

end no_function_satisfies_conditions_l3107_310772


namespace papi_calot_plants_l3107_310702

/-- The number of plants Papi Calot needs to buy -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Proof that Papi Calot needs to buy 141 plants -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end papi_calot_plants_l3107_310702


namespace solve_for_a_l3107_310781

def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

def B : Set ℝ := {3}

theorem solve_for_a (a b : ℝ) (h : A a b = B) : a = -6 := by
  sorry

end solve_for_a_l3107_310781


namespace point_on_line_with_sum_distance_l3107_310760

-- Define the line l
def Line : Type := ℝ → Prop

-- Define the concept of a point being on the same side of a line
def SameSide (l : Line) (A B : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define what it means for a point to be on a line
def OnLine (X : ℝ × ℝ) (l : Line) : Prop := sorry

-- Theorem statement
theorem point_on_line_with_sum_distance 
  (l : Line) (A B : ℝ × ℝ) (a : ℝ) 
  (h1 : SameSide l A B) (h2 : a > 0) : 
  ∃ X : ℝ × ℝ, OnLine X l ∧ distance A X + distance X B = a := by
  sorry

end point_on_line_with_sum_distance_l3107_310760


namespace derivative_of_even_function_l3107_310788

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f is even
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem derivative_of_even_function (hf : IsEven f) :
  ∀ x, (deriv f) (-x) = -(deriv f x) :=
sorry

end derivative_of_even_function_l3107_310788


namespace meaningful_fraction_range_l3107_310762

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end meaningful_fraction_range_l3107_310762


namespace cafeteria_bill_theorem_l3107_310792

/-- The total cost of a cafeteria order for three people -/
def cafeteria_cost (coffee_price ice_cream_price cake_price : ℕ) : ℕ :=
  let mell_order := 2 * coffee_price + cake_price
  let friend_order := 2 * coffee_price + cake_price + ice_cream_price
  mell_order + 2 * friend_order

/-- Theorem stating the total cost for Mell and her friends' cafeteria order -/
theorem cafeteria_bill_theorem :
  cafeteria_cost 4 3 7 = 51 :=
by
  sorry

end cafeteria_bill_theorem_l3107_310792


namespace population_change_proof_l3107_310719

-- Define the initial population
def initial_population : ℕ := 4518

-- Define the sequence of population changes
def population_after_bombardment (p : ℕ) : ℕ := (p * 95) / 100
def population_after_migration (p : ℕ) : ℕ := (p * 80) / 100
def population_after_return (p : ℕ) : ℕ := (p * 115) / 100
def population_after_flood (p : ℕ) : ℕ := (p * 90) / 100

-- Define the final population
def final_population : ℕ := 3553

-- Theorem statement
theorem population_change_proof :
  population_after_flood
    (population_after_return
      (population_after_migration
        (population_after_bombardment initial_population)))
  = final_population := by sorry

end population_change_proof_l3107_310719


namespace order_of_values_l3107_310713

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is monotonically increasing on [0, +∞) if
    for all a, b ≥ 0, a < b implies f(a) < f(b) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 ≤ a → 0 ≤ b → a < b → f a < f b

theorem order_of_values (f : ℝ → ℝ) 
    (h_even : EvenFunction f) 
    (h_mono : MonoIncreasing f) :
    f (-π) > f 3 ∧ f 3 > f (-2) := by
  sorry

end order_of_values_l3107_310713


namespace prove_a_value_l3107_310758

-- Define the operation for integers
def star_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- Theorem statement
theorem prove_a_value (h : star_op 21 9 = 160) : 21 = 21 := by
  sorry

end prove_a_value_l3107_310758


namespace three_numbers_sum_l3107_310732

theorem three_numbers_sum (a b c : ℝ) : 
  b = 2 * a ∧ c = 3 * a ∧ a^2 + b^2 + c^2 = 2744 → a + b + c = 84 := by
sorry

end three_numbers_sum_l3107_310732


namespace quadratic_function_inequality_l3107_310774

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 3) 
  (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ = 1 - a) : 
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
sorry

end quadratic_function_inequality_l3107_310774


namespace fixed_point_of_parabolas_unique_fixed_point_l3107_310780

/-- The fixed point of a family of parabolas -/
theorem fixed_point_of_parabolas (t : ℝ) :
  let f (x : ℝ) := 5 * x^2 + 4 * t * x - 3 * t
  f (3/4) = 45/16 := by sorry

/-- The uniqueness of the fixed point -/
theorem unique_fixed_point (t₁ t₂ : ℝ) (x : ℝ) :
  let f₁ (x : ℝ) := 5 * x^2 + 4 * t₁ * x - 3 * t₁
  let f₂ (x : ℝ) := 5 * x^2 + 4 * t₂ * x - 3 * t₂
  f₁ x = f₂ x → x = 3/4 := by sorry

end fixed_point_of_parabolas_unique_fixed_point_l3107_310780


namespace grocery_store_costs_l3107_310722

theorem grocery_store_costs (total_costs delivery_fraction orders_cost : ℚ)
  (h1 : total_costs = 4000)
  (h2 : orders_cost = 1800)
  (h3 : delivery_fraction = 1/4) :
  let remaining_after_orders := total_costs - orders_cost
  let delivery_cost := delivery_fraction * remaining_after_orders
  let salary_cost := remaining_after_orders - delivery_cost
  salary_cost / total_costs = 33/80 := by
sorry

end grocery_store_costs_l3107_310722


namespace constant_term_expansion_l3107_310777

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (5 * x + 1 / (3 * x)) ^ 8
  ∃ (p q : ℝ → ℝ), expansion = p x + (43750 / 81) + q x ∧ 
    (∀ y, y ≠ 0 → p y + q y = 0) :=
by sorry

end constant_term_expansion_l3107_310777


namespace sum_of_coefficients_P_l3107_310768

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^9 - 3 * x^6 + 4) - 4 * (x^6 - 5 * x^3 + 6)

-- Theorem stating that the sum of coefficients of P(x) is 7
theorem sum_of_coefficients_P : (P 1) = 7 := by
  sorry

end sum_of_coefficients_P_l3107_310768


namespace circle_equation_l3107_310755

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := 3*x + 4*y + 2 = 0

-- Define the general circle
def general_circle (x y b r : ℝ) : Prop := (x - 1)^2 + (y - b)^2 = r^2

-- Define the specific circle we want to prove
def specific_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- State the theorem
theorem circle_equation :
  ∀ (b r : ℝ),
  (∀ x y, parabola x y → x = 1 ∧ y = 0) →  -- Focus of parabola is (1, 0)
  (∀ x y, line x y → general_circle x y b r) →  -- Line is tangent to circle
  (∀ x y, specific_circle x y) :=
by sorry

end circle_equation_l3107_310755


namespace carla_cards_theorem_l3107_310725

/-- A structure representing a card with two numbers -/
structure Card where
  visible : ℕ
  hidden : ℕ

/-- The setup of Carla's cards -/
def carla_cards : Card × Card :=
  ⟨⟨37, 0⟩, ⟨53, 0⟩⟩

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Theorem stating the properties of Carla's card setup and the result -/
theorem carla_cards_theorem (cards : Card × Card) : 
  cards = carla_cards →
  (∃ p₁ p₂ : ℕ, 
    is_prime p₁ ∧ 
    is_prime p₂ ∧ 
    p₁ ≠ p₂ ∧
    cards.1.visible + p₁ = cards.2.visible + p₂ ∧
    (p₁ + p₂) / 2 = 11) := by
  sorry

#check carla_cards_theorem

end carla_cards_theorem_l3107_310725


namespace proportional_function_k_value_l3107_310769

/-- A proportional function passing through a specific point -/
def proportional_function (k : ℝ) (x : ℝ) : ℝ := k * x

theorem proportional_function_k_value :
  ∀ k : ℝ,
  k ≠ 0 →
  proportional_function k 3 = -6 →
  k = -2 := by
sorry

end proportional_function_k_value_l3107_310769


namespace N_subset_M_l3107_310704

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = k / 2 + 1 / 3}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k + 1 / 3}

-- Theorem statement
theorem N_subset_M : N ⊆ M := by
  sorry

end N_subset_M_l3107_310704


namespace michael_passes_donovan_l3107_310753

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℝ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℝ := 40

/-- The number of laps Michael needs to complete to pass Donovan -/
def laps_to_pass : ℕ := 9

theorem michael_passes_donovan :
  (laps_to_pass : ℝ) * michael_lap_time = (laps_to_pass - 1 : ℝ) * donovan_lap_time :=
sorry

end michael_passes_donovan_l3107_310753


namespace product_division_result_l3107_310734

theorem product_division_result : (1.6 * 0.5) / 1 = 0.8 := by
  sorry

end product_division_result_l3107_310734


namespace tylers_dogs_l3107_310743

theorem tylers_dogs (puppies_per_dog : ℕ) (total_puppies : ℕ) (initial_dogs : ℕ) : 
  puppies_per_dog = 5 → 
  total_puppies = 75 → 
  initial_dogs * puppies_per_dog = total_puppies → 
  initial_dogs = 15 := by
sorry

end tylers_dogs_l3107_310743


namespace double_box_11_l3107_310776

def box_sum (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem double_box_11 : box_sum (box_sum 11) = 28 := by
  sorry

end double_box_11_l3107_310776


namespace hexagon_area_from_triangle_l3107_310761

/-- Given an equilateral triangle and a regular hexagon with equal perimeters,
    if the area of the triangle is β, then the area of the hexagon is (3/2) * β. -/
theorem hexagon_area_from_triangle (β : ℝ) :
  ∀ (x y : ℝ),
  x > 0 → y > 0 →
  (3 * x = 6 * y) →  -- Equal perimeters
  (β = Real.sqrt 3 / 4 * x^2) →  -- Area of equilateral triangle
  ∃ (γ : ℝ), γ = 3 * Real.sqrt 3 / 2 * y^2 ∧ γ = 3/2 * β := by
  sorry

end hexagon_area_from_triangle_l3107_310761


namespace large_rectangle_ratio_l3107_310751

/-- Represents the side length of a square in the arrangement -/
def square_side : ℝ := sorry

/-- Represents the length of the large rectangle -/
def large_rectangle_length : ℝ := 3 * square_side

/-- Represents the width of the large rectangle -/
def large_rectangle_width : ℝ := 3 * square_side

/-- Represents the length of the smaller rectangle -/
def small_rectangle_length : ℝ := 3 * square_side

/-- Represents the width of the smaller rectangle -/
def small_rectangle_width : ℝ := square_side

theorem large_rectangle_ratio :
  large_rectangle_length / large_rectangle_width = 3 := by sorry

end large_rectangle_ratio_l3107_310751


namespace P_on_y_axis_P_in_first_quadrant_with_distance_condition_l3107_310782

-- Define point P
def P (m : ℝ) : ℝ × ℝ := (8 - 2*m, m + 1)

-- Part 1: P lies on y-axis
theorem P_on_y_axis (m : ℝ) : 
  P m = (0, m + 1) → m = 4 := by sorry

-- Part 2: P in first quadrant with distance condition
theorem P_in_first_quadrant_with_distance_condition (m : ℝ) :
  (8 - 2*m > 0 ∧ m + 1 > 0) ∧ (m + 1 = 2*(8 - 2*m)) → P m = (2, 4) := by sorry

end P_on_y_axis_P_in_first_quadrant_with_distance_condition_l3107_310782


namespace gcd_digits_bound_l3107_310779

theorem gcd_digits_bound (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) →
  Nat.gcd a b < 10^3 := by
  sorry

end gcd_digits_bound_l3107_310779


namespace unique_solution_l3107_310717

-- Define Θ as a natural number
variable (Θ : ℕ)

-- Define the condition that Θ is a single digit
def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

-- Define the two-digit number 4Θ
def four_Θ (Θ : ℕ) : ℕ := 40 + Θ

-- State the theorem
theorem unique_solution :
  (630 / Θ = four_Θ Θ + 2 * Θ) ∧ 
  (is_single_digit Θ) →
  Θ = 9 :=
sorry

end unique_solution_l3107_310717


namespace triangle_analogous_to_tetrahedron_l3107_310778

/-- Represents geometric objects -/
inductive GeometricObject
  | Quadrilateral
  | Pyramid
  | Triangle
  | Prism
  | Tetrahedron

/-- Defines the concept of analogous objects based on shared properties -/
def are_analogous (a b : GeometricObject) : Prop :=
  ∃ (property : GeometricObject → Prop), property a ∧ property b

/-- Theorem stating that a triangle is analogous to a tetrahedron -/
theorem triangle_analogous_to_tetrahedron :
  are_analogous GeometricObject.Triangle GeometricObject.Tetrahedron :=
sorry

end triangle_analogous_to_tetrahedron_l3107_310778


namespace complex_magnitude_equals_seven_l3107_310752

theorem complex_magnitude_equals_seven (t : ℝ) (h1 : t > 0) :
  Complex.abs (3 + t * Complex.I) = 7 → t = 2 * Real.sqrt 10 := by
  sorry

end complex_magnitude_equals_seven_l3107_310752


namespace constant_function_not_decreasing_l3107_310746

def f : ℝ → ℝ := fun _ ↦ 2

theorem constant_function_not_decreasing :
  ¬∃ (a b : ℝ), a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x := by
  sorry

end constant_function_not_decreasing_l3107_310746


namespace picnic_total_attendance_l3107_310721

/-- The number of persons at a picnic -/
def picnic_attendance (men women adults children : ℕ) : Prop :=
  (men = women + 20) ∧ 
  (adults = children + 20) ∧ 
  (men = 65) ∧
  (men + women + children = 200)

/-- Theorem stating the total number of persons at the picnic -/
theorem picnic_total_attendance :
  ∃ (men women adults children : ℕ),
    picnic_attendance men women adults children :=
by
  sorry

end picnic_total_attendance_l3107_310721


namespace increased_chickens_sum_l3107_310742

/-- The number of increased chickens since the beginning -/
def increased_chickens (original : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  first_day + second_day

/-- Theorem stating that the number of increased chickens is the sum of chickens brought on the first and second day -/
theorem increased_chickens_sum (original : ℕ) (first_day : ℕ) (second_day : ℕ) :
  increased_chickens original first_day second_day = first_day + second_day :=
by sorry

#eval increased_chickens 45 18 12

end increased_chickens_sum_l3107_310742


namespace daisy_spending_difference_l3107_310797

def breakfast_muffin1_price : ℚ := 2
def breakfast_muffin2_price : ℚ := 3
def breakfast_coffee1_price : ℚ := 4
def breakfast_coffee2_discount : ℚ := 0.5
def lunch_soup_price : ℚ := 3.75
def lunch_salad_price : ℚ := 5.75
def lunch_lemonade_price : ℚ := 1
def lunch_service_charge_percent : ℚ := 10

def breakfast_total : ℚ := breakfast_muffin1_price + breakfast_muffin2_price + breakfast_coffee1_price + (breakfast_coffee1_price - breakfast_coffee2_discount)

def lunch_subtotal : ℚ := lunch_soup_price + lunch_salad_price + lunch_lemonade_price

def lunch_total : ℚ := lunch_subtotal + (lunch_subtotal * lunch_service_charge_percent / 100)

theorem daisy_spending_difference : lunch_total - breakfast_total = -0.95 := by
  sorry

end daisy_spending_difference_l3107_310797


namespace book_area_l3107_310707

/-- The area of a rectangle with length 2 inches and width 3 inches is 6 square inches. -/
theorem book_area : 
  ∀ (length width area : ℝ), 
    length = 2 → 
    width = 3 → 
    area = length * width → 
    area = 6 := by
  sorry

end book_area_l3107_310707


namespace square_of_two_digit_is_68_l3107_310738

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ := n / 1000

def last_digit (n : ℕ) : ℕ := n % 10

def middle_digits_sum (n : ℕ) : ℕ := (n / 100 % 10) + (n / 10 % 10)

theorem square_of_two_digit_is_68 (n : ℕ) (h1 : is_four_digit n) 
  (h2 : first_digit n = last_digit n)
  (h3 : first_digit n + last_digit n = middle_digits_sum n)
  (h4 : ∃ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ m * m = n) :
  ∃ m : ℕ, m = 68 ∧ m * m = n := by
sorry

end square_of_two_digit_is_68_l3107_310738


namespace exists_player_reaching_all_l3107_310706

/-- Represents a tournament where every player plays against every other player once with no draws -/
structure Tournament (α : Type) :=
  (players : Set α)
  (defeated : α → α → Prop)
  (complete : ∀ a b : α, a ≠ b → (defeated a b ∨ defeated b a))
  (irreflexive : ∀ a : α, ¬ defeated a a)

/-- A player can reach another player within two steps of the defeated relation -/
def can_reach_in_two_steps {α : Type} (t : Tournament α) (a b : α) : Prop :=
  t.defeated a b ∨ ∃ c, t.defeated a c ∧ t.defeated c b

/-- The main theorem: there exists a player who can reach all others within two steps -/
theorem exists_player_reaching_all {α : Type} (t : Tournament α) :
  ∃ a : α, ∀ b : α, b ∈ t.players → a ≠ b → can_reach_in_two_steps t a b :=
sorry

end exists_player_reaching_all_l3107_310706


namespace log_inequality_condition_l3107_310799

theorem log_inequality_condition (a b : ℝ) : 
  (∀ a b, Real.log a > Real.log b → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) :=
sorry

end log_inequality_condition_l3107_310799


namespace square_root_of_nine_l3107_310791

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end square_root_of_nine_l3107_310791


namespace puppies_adopted_per_day_l3107_310750

theorem puppies_adopted_per_day :
  ∀ (initial_puppies additional_puppies total_days : ℕ),
    initial_puppies = 3 →
    additional_puppies = 3 →
    total_days = 2 →
    (initial_puppies + additional_puppies) / total_days = 3 :=
by
  sorry

#check puppies_adopted_per_day

end puppies_adopted_per_day_l3107_310750


namespace smallest_perimeter_circle_circle_center_on_line_l3107_310739

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define the general equation of a circle
def circle_general_eq (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Define the standard equation of a circle
def circle_standard_eq (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem for the circle with smallest perimeter
theorem smallest_perimeter_circle :
  ∃ (x y : ℝ), x^2 + y^2 - 2*y - 9 = 0 ∧
  circle_general_eq x y 0 1 (5 : ℝ) ∧
  (∀ (a b r : ℝ), circle_general_eq A.1 A.2 a b r → 
   circle_general_eq B.1 B.2 a b r → 
   r^2 ≥ 10) := by sorry

-- Theorem for the circle with center on the given line
theorem circle_center_on_line :
  ∃ (x y : ℝ), (x - 3)^2 + (y - 2)^2 = 20 ∧
  circle_standard_eq x y 3 2 (2 * Real.sqrt 5) ∧
  line_eq 3 2 := by sorry

end smallest_perimeter_circle_circle_center_on_line_l3107_310739


namespace gcd_1230_990_l3107_310775

theorem gcd_1230_990 : Nat.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l3107_310775


namespace arctan_sum_three_seven_l3107_310754

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end arctan_sum_three_seven_l3107_310754


namespace battle_station_staffing_l3107_310749

theorem battle_station_staffing (n m : ℕ) (h1 : n = 20) (h2 : m = 5) :
  (n - 1).factorial / (n - m).factorial = 930240 := by
  sorry

end battle_station_staffing_l3107_310749


namespace union_of_sets_l3107_310786

def setA : Set ℝ := {x | x + 2 > 0}
def setB : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem union_of_sets : setA ∪ setB = Set.Ioi (-2) := by sorry

end union_of_sets_l3107_310786


namespace chloe_winter_clothing_l3107_310748

/-- The number of boxes Chloe has -/
def num_boxes : ℕ := 4

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of winter clothing pieces Chloe has -/
def total_pieces : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem chloe_winter_clothing :
  total_pieces = 32 :=
by sorry

end chloe_winter_clothing_l3107_310748


namespace f_minimum_l3107_310726

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- State the theorem
theorem f_minimum (a b : ℝ) (h : 1 / (2 * a) + 2 / b = 1) :
  ∀ x : ℝ, f x a b ≥ 9/2 :=
by sorry

end f_minimum_l3107_310726


namespace enlarged_poster_height_l3107_310747

-- Define the original poster dimensions
def original_width : ℚ := 3
def original_height : ℚ := 2

-- Define the new width
def new_width : ℚ := 12

-- Define the function to calculate the new height
def calculate_new_height (ow oh nw : ℚ) : ℚ :=
  (nw / ow) * oh

-- Theorem statement
theorem enlarged_poster_height :
  calculate_new_height original_width original_height new_width = 8 := by
  sorry

end enlarged_poster_height_l3107_310747


namespace rectangle_area_perimeter_sum_l3107_310764

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  let A := (a : ℕ) * b
  let P := 2 * ((a : ℕ) + b)
  A + P ≠ 102 :=
by sorry

end rectangle_area_perimeter_sum_l3107_310764


namespace emilias_blueberries_l3107_310701

/-- The number of cartons of berries Emilia needs in total -/
def total_needed : ℕ := 42

/-- The number of cartons of strawberries Emilia has -/
def strawberries : ℕ := 2

/-- The number of cartons of berries Emilia buys at the supermarket -/
def bought : ℕ := 33

/-- The number of cartons of blueberries in Emilia's cupboard -/
def blueberries : ℕ := total_needed - (strawberries + bought)

theorem emilias_blueberries : blueberries = 7 := by
  sorry

end emilias_blueberries_l3107_310701


namespace same_color_probability_l3107_310796

/-- The probability of drawing two balls of the same color from a bag containing
    8 green balls, 5 red balls, and 7 blue balls, with replacement. -/
theorem same_color_probability :
  let total_balls : ℕ := 8 + 5 + 7
  let p_green : ℚ := 8 / total_balls
  let p_red : ℚ := 5 / total_balls
  let p_blue : ℚ := 7 / total_balls
  let p_same_color : ℚ := p_green ^ 2 + p_red ^ 2 + p_blue ^ 2
  p_same_color = 117 / 200 := by
  sorry

end same_color_probability_l3107_310796


namespace monitor_student_ratio_l3107_310714

/-- The ratio of monitors to students in a lunchroom --/
theorem monitor_student_ratio :
  ∀ (S : ℕ) (G B : ℝ),
    G = 0.4 * S →
    B = 0.6 * S →
    2 * G + B = 168 →
    (8 : ℝ) / S = 1 / 15 :=
by sorry

end monitor_student_ratio_l3107_310714


namespace train_crossing_time_l3107_310720

/-- Time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 100 →
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * 1000 / 3600)) = 6 := by
  sorry

end train_crossing_time_l3107_310720


namespace average_sale_proof_l3107_310736

def sales : List ℝ := [5266, 5768, 5922, 5678, 6029]
def required_sale : ℝ := 4937

theorem average_sale_proof :
  (sales.sum + required_sale) / 6 = 5600 := by
  sorry

end average_sale_proof_l3107_310736


namespace one_minus_repeating_eight_eq_one_ninth_l3107_310740

/-- The repeating decimal 0.overline{8} -/
def repeating_eight : ℚ := 8/9

/-- Theorem stating that 1 minus the repeating decimal 0.overline{8} equals 1/9 -/
theorem one_minus_repeating_eight_eq_one_ninth : 1 - repeating_eight = 1/9 := by
  sorry

end one_minus_repeating_eight_eq_one_ninth_l3107_310740


namespace zero_points_inequality_l3107_310703

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - (a / 2) * Real.log x

theorem zero_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x > 0 → f a x = 0 → x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  1 < x₁ ∧ x₁ < a ∧ a < x₂ ∧ x₂ < a^2 :=
by sorry

end zero_points_inequality_l3107_310703
