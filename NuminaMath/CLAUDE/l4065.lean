import Mathlib

namespace fraction_simplification_l4065_406579

theorem fraction_simplification : (5 * (8 + 2)) / 10 = 5 := by
  sorry

end fraction_simplification_l4065_406579


namespace hannah_games_played_l4065_406572

def total_points : ℕ := 312
def average_points : ℕ := 13

theorem hannah_games_played : 
  (total_points / average_points : ℕ) = 24 :=
by sorry

end hannah_games_played_l4065_406572


namespace system_solution_range_l4065_406577

theorem system_solution_range (x y m : ℝ) : 
  (3 * x + y = 1 + 3 * m) →
  (x + 3 * y = 1 - m) →
  (x + y > 0) →
  (m > -1) := by
sorry

end system_solution_range_l4065_406577


namespace min_m_plus_n_range_of_a_l4065_406522

-- Part I
theorem min_m_plus_n (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = |x + 1| + (1/2) * |2*x - 1|) →
  (m > 0 ∧ n > 0) →
  (∀ x, f x ≥ 1/m + 1/n) →
  m + n ≥ 8/3 :=
sorry

-- Part II
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = |x + 1| + a * |2*x - 1|) →
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ |x - 2|) →
  a ≥ 1 :=
sorry

end min_m_plus_n_range_of_a_l4065_406522


namespace arithmetic_sequence_property_l4065_406591

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l4065_406591


namespace tom_hiking_probability_l4065_406564

theorem tom_hiking_probability (p_fog : ℝ) (p_hike_foggy : ℝ) (p_hike_clear : ℝ)
  (h_fog : p_fog = 0.5)
  (h_hike_foggy : p_hike_foggy = 0.3)
  (h_hike_clear : p_hike_clear = 0.9) :
  p_fog * p_hike_foggy + (1 - p_fog) * p_hike_clear = 0.6 := by
  sorry

#check tom_hiking_probability

end tom_hiking_probability_l4065_406564


namespace circus_ticket_cost_l4065_406548

theorem circus_ticket_cost (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_cost : ℕ) : 
  cost_per_ticket = 44 → num_tickets = 7 → total_cost = cost_per_ticket * num_tickets →
  total_cost = 308 := by
sorry

end circus_ticket_cost_l4065_406548


namespace intersection_kth_element_l4065_406503

-- Define set A
def A : Set ℕ := {n | ∃ m : ℕ, n = m * (m + 1)}

-- Define set B
def B : Set ℕ := {n | ∃ m : ℕ, n = 3 * m - 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℕ := A ∩ B

-- Define the kth element of the intersection
def a (k : ℕ) : ℕ := 9 * k^2 - 9 * k + 2

-- Theorem statement
theorem intersection_kth_element (k : ℕ) : 
  a k ∈ A_intersect_B ∧ 
  (∀ n ∈ A_intersect_B, n < a k → 
    ∃ j < k, n = a j) ∧
  (∀ n ∈ A_intersect_B, n ≠ a k → 
    ∃ j ≠ k, n = a j) :=
sorry

end intersection_kth_element_l4065_406503


namespace square_of_linear_cyclic_l4065_406598

variable (a b c A B C : ℝ)

/-- Two linear polynomials sum to a square of a linear polynomial iff their coefficients satisfy this condition -/
def is_square_of_linear (α β γ δ : ℝ) : Prop :=
  α * δ = β * γ

/-- The main theorem: if two pairs of expressions are squares of linear polynomials, 
    then the third pair is also a square of a linear polynomial -/
theorem square_of_linear_cyclic 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0)
  (h1 : is_square_of_linear a b A B)
  (h2 : is_square_of_linear b c B C) :
  is_square_of_linear c a C A :=
sorry

end square_of_linear_cyclic_l4065_406598


namespace parallel_line_slope_l4065_406593

/-- Given a line with equation 3x - 6y = 12, prove that its slope (and the slope of any parallel line) is 1/2. -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → (∃ m b : ℝ, y = m * x + b ∧ m = 1/2) :=
by sorry

end parallel_line_slope_l4065_406593


namespace samuel_apple_ratio_l4065_406563

/-- Prove that the ratio of apples Samuel ate to the total number of apples he bought is 1:2 -/
theorem samuel_apple_ratio :
  let bonnie_apples : ℕ := 8
  let samuel_extra_apples : ℕ := 20
  let samuel_total_apples : ℕ := bonnie_apples + samuel_extra_apples
  let samuel_pie_apples : ℕ := samuel_total_apples / 7
  let samuel_left_apples : ℕ := 10
  let samuel_eaten_apples : ℕ := samuel_total_apples - samuel_pie_apples - samuel_left_apples
  (samuel_eaten_apples : ℚ) / (samuel_total_apples : ℚ) = 1 / 2 := by
  sorry

end samuel_apple_ratio_l4065_406563


namespace polynomial_product_expansion_l4065_406536

theorem polynomial_product_expansion (x : ℝ) : 
  (5 * x + 3) * (6 * x^2 + 2) = 30 * x^3 + 18 * x^2 + 10 * x + 6 := by
  sorry

end polynomial_product_expansion_l4065_406536


namespace hcf_36_84_l4065_406557

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_36_84_l4065_406557


namespace stones_kept_as_favorite_l4065_406546

theorem stones_kept_as_favorite (original_stones sent_away_stones : ℕ) 
  (h1 : original_stones = 78) 
  (h2 : sent_away_stones = 63) : 
  original_stones - sent_away_stones = 15 := by
  sorry

end stones_kept_as_favorite_l4065_406546


namespace sum_rational_irrational_not_rational_l4065_406583

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sum_rational_irrational_not_rational :
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r + i) :=
by
  sorry

end sum_rational_irrational_not_rational_l4065_406583


namespace f_positive_iff_l4065_406545

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 3 := by
  sorry

end f_positive_iff_l4065_406545


namespace quadratic_roots_sum_squares_minus_product_l4065_406515

theorem quadratic_roots_sum_squares_minus_product (m n : ℝ) : 
  m^2 - 5*m - 2 = 0 → n^2 - 5*n - 2 = 0 → m^2 + n^2 - m*n = 31 := by
  sorry

end quadratic_roots_sum_squares_minus_product_l4065_406515


namespace arithmetic_sequence_properties_l4065_406554

/-- An arithmetic sequence with general term a_n = 4n - 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

/-- The first term of the sequence -/
def first_term : ℤ := arithmetic_sequence 1

/-- The second term of the sequence -/
def second_term : ℤ := arithmetic_sequence 2

/-- The common difference of the sequence -/
def common_difference : ℤ := second_term - first_term

theorem arithmetic_sequence_properties :
  first_term = 1 ∧ common_difference = 4 := by sorry

end arithmetic_sequence_properties_l4065_406554


namespace pole_length_problem_l4065_406569

theorem pole_length_problem (original_length : ℝ) (cut_length : ℝ) : 
  cut_length = 0.7 * original_length →
  cut_length = 14 →
  original_length = 20 := by
sorry

end pole_length_problem_l4065_406569


namespace smallest_group_sample_size_l4065_406587

def stratified_sampling (total_sample_size : ℕ) (group_ratios : List ℕ) : List ℕ :=
  let total_ratio := group_ratios.sum
  group_ratios.map (λ ratio => (total_sample_size * ratio) / total_ratio)

theorem smallest_group_sample_size 
  (total_sample_size : ℕ) 
  (group_ratios : List ℕ) :
  total_sample_size = 20 →
  group_ratios = [5, 4, 1] →
  (stratified_sampling total_sample_size group_ratios).getLast! = 2 :=
by
  sorry

#eval stratified_sampling 20 [5, 4, 1]

end smallest_group_sample_size_l4065_406587


namespace emily_necklaces_l4065_406517

/-- Given that Emily used a total of 18 beads and each necklace requires 3 beads,
    prove that the number of necklaces she made is 6. -/
theorem emily_necklaces :
  let total_beads : ℕ := 18
  let beads_per_necklace : ℕ := 3
  let necklaces_made : ℕ := total_beads / beads_per_necklace
  necklaces_made = 6 := by
  sorry

end emily_necklaces_l4065_406517


namespace mass_of_man_is_60kg_l4065_406558

/-- The mass of a man who causes a boat to sink by a certain depth --/
def mass_of_man (boat_length boat_breadth sink_depth water_density : Real) : Real :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg given the specific conditions --/
theorem mass_of_man_is_60kg : 
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

#eval mass_of_man 3 2 0.01 1000

end mass_of_man_is_60kg_l4065_406558


namespace set_equality_l4065_406584

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l4065_406584


namespace sqrt_sum_equals_thirteen_sixths_l4065_406576

theorem sqrt_sum_equals_thirteen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (4 / 9) = 13 / 6 := by
  sorry

end sqrt_sum_equals_thirteen_sixths_l4065_406576


namespace line_equation_given_ellipse_midpoint_l4065_406523

/-- The equation of a line that intersects an ellipse, given the midpoint of the intersection -/
theorem line_equation_given_ellipse_midpoint (x y : ℝ → ℝ) (l : Set (ℝ × ℝ)) :
  (∀ t, (x t)^2 / 36 + (y t)^2 / 9 = 1) →  -- Ellipse equation
  (∃ t₁ t₂, (x t₁, y t₁) ∈ l ∧ (x t₂, y t₂) ∈ l ∧ t₁ ≠ t₂) →  -- Line intersects ellipse at two points
  ((x t₁ + x t₂) / 2 = 4 ∧ (y t₁ + y t₂) / 2 = 2) →  -- Midpoint is (4,2)
  (∀ p, p ∈ l ↔ p.1 + 2 * p.2 - 8 = 0) :=  -- Line equation
by sorry

end line_equation_given_ellipse_midpoint_l4065_406523


namespace product_of_eights_place_values_l4065_406521

/-- The place value of a digit in a decimal number -/
def place_value (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (10 : ℚ) ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 780.38

/-- The theorem stating that the product of place values of two 8's in 780.38 is 6.4 -/
theorem product_of_eights_place_values :
  (place_value 8 1) * (place_value 8 (-2)) = 6.4 := by sorry

end product_of_eights_place_values_l4065_406521


namespace parameterized_line_solution_l4065_406580

/-- The line y = 4x - 7 parameterized by (x, y) = (s, -3) + t(3, m) -/
def parameterized_line (s m t : ℝ) : ℝ × ℝ :=
  (s + 3*t, -3 + m*t)

/-- The line y = 4x - 7 -/
def line (x y : ℝ) : Prop :=
  y = 4*x - 7

theorem parameterized_line_solution :
  ∃ (s m : ℝ), ∀ (t : ℝ),
    let (x, y) := parameterized_line s m t
    line x y ∧ s = 1 ∧ m = 12 := by
  sorry

end parameterized_line_solution_l4065_406580


namespace simplification_fraction_l4065_406543

theorem simplification_fraction (k : ℝ) :
  ∃ (c d : ℤ), (6 * k + 12 + 3) / 3 = c * k + d ∧ (c : ℚ) / d = 2 / 5 := by
  sorry

end simplification_fraction_l4065_406543


namespace circle_through_points_equation_l4065_406537

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The circle passing through three given points -/
def CircleThroughThreePoints (A B C : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (center.x - A.x)^2 + (center.y - A.y)^2 = radius^2 ∧
    (center.x - B.x)^2 + (center.y - B.y)^2 = radius^2 ∧
    (center.x - C.x)^2 + (center.y - C.y)^2 = radius^2

/-- The standard equation of a circle -/
def StandardCircleEquation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem circle_through_points_equation :
  let A : Point := ⟨1, 12⟩
  let B : Point := ⟨7, 10⟩
  let C : Point := ⟨-9, 2⟩
  CircleThroughThreePoints A B C →
  ∃ (x y : ℝ), StandardCircleEquation ⟨1, 2⟩ 10 x y :=
by sorry

end circle_through_points_equation_l4065_406537


namespace seashells_given_to_sam_l4065_406505

/-- Given that Joan found a certain number of seashells and has some left after giving some to Sam,
    prove that the number of seashells given to Sam is the difference between the initial and remaining amounts. -/
theorem seashells_given_to_sam 
  (initial : ℕ) 
  (remaining : ℕ) 
  (h1 : initial = 70) 
  (h2 : remaining = 27) 
  (h3 : remaining < initial) : 
  initial - remaining = 43 := by
sorry

end seashells_given_to_sam_l4065_406505


namespace coin_division_sum_equals_pairs_l4065_406525

/-- Represents the process of dividing coins into piles --/
def CoinDivisionProcess : Type := List (Nat × Nat)

/-- The number of coins --/
def n : Nat := 25

/-- Calculates the sum of products for a given division process --/
def sum_of_products (process : CoinDivisionProcess) : Nat :=
  process.foldl (fun sum pair => sum + pair.1 * pair.2) 0

/-- Represents all possible division processes for n coins --/
def all_division_processes (n : Nat) : Set CoinDivisionProcess :=
  sorry

/-- Theorem stating that the sum of products equals the number of pairs of coins --/
theorem coin_division_sum_equals_pairs :
  ∀ (process : CoinDivisionProcess),
    process ∈ all_division_processes n →
    sum_of_products process = n.choose 2 := by
  sorry

end coin_division_sum_equals_pairs_l4065_406525


namespace cos_angle_between_vectors_l4065_406529

/-- Given two vectors in R², prove that the cosine of the angle between them is 4/5 -/
theorem cos_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (4, 2) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 4/5 := by
sorry

end cos_angle_between_vectors_l4065_406529


namespace prob_3_red_in_5_draws_eq_8_81_l4065_406511

/-- The probability of drawing a red ball from the bag -/
def prob_red : ℚ := 1 / 3

/-- The probability of drawing a white ball from the bag -/
def prob_white : ℚ := 2 / 3

/-- The number of ways to choose 2 red balls from 4 draws -/
def ways_to_choose_2_from_4 : ℕ := 6

/-- The probability of drawing exactly 3 red balls in 5 draws, with the last draw being red -/
def prob_3_red_in_5_draws : ℚ :=
  ways_to_choose_2_from_4 * prob_red^2 * prob_white^2 * prob_red

theorem prob_3_red_in_5_draws_eq_8_81 : 
  prob_3_red_in_5_draws = 8 / 81 := by sorry

end prob_3_red_in_5_draws_eq_8_81_l4065_406511


namespace star_property_l4065_406539

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.two
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.four
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.one
  | Element.two, Element.four => Element.three
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.two
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.three
  | Element.four, Element.three => Element.two
  | Element.four, Element.four => Element.one

theorem star_property : 
  star (star Element.two Element.four) (star Element.one Element.three) = Element.four := by
  sorry

end star_property_l4065_406539


namespace encryption_of_2568_l4065_406504

def encrypt_digit (d : Nat) : Nat :=
  (d^3 + 1) % 10

def encrypt_number (n : List Nat) : List Nat :=
  n.map encrypt_digit

theorem encryption_of_2568 :
  encrypt_number [2, 5, 6, 8] = [9, 6, 7, 3] := by
  sorry

end encryption_of_2568_l4065_406504


namespace cello_practice_time_l4065_406512

/-- Calculates the remaining practice time in minutes for a cellist -/
theorem cello_practice_time (total_hours : ℝ) (daily_minutes : ℕ) (practice_days : ℕ) :
  total_hours = 7.5 ∧ daily_minutes = 86 ∧ practice_days = 2 →
  (total_hours * 60 : ℝ) - (daily_minutes * practice_days : ℕ) = 278 := by
  sorry

#check cello_practice_time

end cello_practice_time_l4065_406512


namespace fraction_value_l4065_406594

theorem fraction_value (x : ℤ) : 
  (∃ (n : ℕ+), (2 : ℚ) / (x + 1 : ℚ) = n) → x = 0 ∨ x = 1 := by
  sorry

end fraction_value_l4065_406594


namespace root_in_interval_l4065_406574

def f (x : ℝ) := 2*x + x - 2

theorem root_in_interval :
  Continuous f ∧ f 0 < 0 ∧ f 1 > 0 → ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
by
  sorry

end root_in_interval_l4065_406574


namespace sqrt_equation_solution_l4065_406575

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 13) = 11 → x = 108 := by
  sorry

end sqrt_equation_solution_l4065_406575


namespace rebecca_groups_l4065_406532

/-- The number of eggs Rebecca has -/
def total_eggs : ℕ := 20

/-- The number of marbles Rebecca has -/
def total_marbles : ℕ := 6

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 5

/-- The number of marbles in each group -/
def marbles_per_group : ℕ := 2

/-- The maximum number of groups that can be created -/
def max_groups : ℕ := min (total_eggs / eggs_per_group) (total_marbles / marbles_per_group)

theorem rebecca_groups : max_groups = 3 := by
  sorry

end rebecca_groups_l4065_406532


namespace average_of_five_numbers_l4065_406561

theorem average_of_five_numbers (total : ℕ) (avg_all : ℚ) (avg_three : ℚ) :
  total = 8 →
  avg_all = 20 →
  avg_three = 33333333333333336 / 1000000000000000 →
  let sum_all := avg_all * total
  let sum_three := avg_three * 3
  let sum_five := sum_all - sum_three
  sum_five / 5 = 12 := by
sorry

#eval 33333333333333336 / 1000000000000000  -- To verify the fraction equals 33.333333333333336

end average_of_five_numbers_l4065_406561


namespace equation_solution_l4065_406533

theorem equation_solution :
  ∀ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 9 := by
  sorry

end equation_solution_l4065_406533


namespace square_49_equals_square_50_minus_99_l4065_406528

theorem square_49_equals_square_50_minus_99 : 49^2 = 50^2 - 99 := by
  sorry

end square_49_equals_square_50_minus_99_l4065_406528


namespace max_inscribed_equilateral_triangle_area_l4065_406589

/-- The maximum area of an equilateral triangle inscribed in a 12 by 15 rectangle -/
theorem max_inscribed_equilateral_triangle_area :
  ∃ (A : ℝ), A = 48 * Real.sqrt 3 ∧
  ∀ (s : ℝ), s > 0 →
    s * Real.sqrt 3 / 2 ≤ 12 →
    s ≤ 15 →
    s * s * Real.sqrt 3 / 4 ≤ A :=
by sorry

end max_inscribed_equilateral_triangle_area_l4065_406589


namespace larger_number_value_l4065_406566

theorem larger_number_value (x y : ℝ) (hx : x = 48) (hdiff : y - x = (1/3) * y) : y = 72 := by
  sorry

end larger_number_value_l4065_406566


namespace father_steps_problem_l4065_406568

/-- Calculates the number of steps taken by Father given the step ratios and total steps of children -/
def father_steps (father_masha_ratio : ℚ) (masha_yasha_ratio : ℚ) (total_children_steps : ℕ) : ℕ :=
  sorry

/-- Theorem stating that Father takes 90 steps given the problem conditions -/
theorem father_steps_problem :
  let father_masha_ratio : ℚ := 3 / 5
  let masha_yasha_ratio : ℚ := 3 / 5
  let total_children_steps : ℕ := 400
  father_steps father_masha_ratio masha_yasha_ratio total_children_steps = 90 := by
  sorry

end father_steps_problem_l4065_406568


namespace parallel_lines_j_value_l4065_406590

/-- Given two points on a line and another line equation, find the j-coordinate of the second point -/
theorem parallel_lines_j_value :
  let line1_point1 : ℝ × ℝ := (5, -6)
  let line1_point2 : ℝ × ℝ := (j, 29)
  let line2_slope : ℝ := 3 / 2
  let line2_equation (x y : ℝ) := 3 * x - 2 * y = 15
  ∀ j : ℝ,
    (line1_point2.2 - line1_point1.2) / (line1_point2.1 - line1_point1.1) = line2_slope →
    j = 85 / 3 :=
by
  sorry

end parallel_lines_j_value_l4065_406590


namespace money_division_l4065_406530

theorem money_division (p q r : ℕ) (total : ℚ) : 
  p + q + r = 22 →  -- ratio sum: 3 + 7 + 12 = 22
  (12 / 22) * total - (7 / 22) * total = 4000 →
  (7 / 22) * total - (3 / 22) * total = 3200 :=
by
  sorry

end money_division_l4065_406530


namespace charlies_share_l4065_406518

theorem charlies_share (total : ℚ) (a b c : ℚ) : 
  total = 10000 →
  a = (1/3) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 6000 := by
sorry

end charlies_share_l4065_406518


namespace gabriel_has_35_boxes_l4065_406507

-- Define the number of boxes for each person
def stan_boxes : ℕ := 120

-- Define relationships between box counts
def joseph_boxes : ℕ := (stan_boxes * 20) / 100
def jules_boxes : ℕ := joseph_boxes + 5
def john_boxes : ℕ := (jules_boxes * 120) / 100
def martin_boxes : ℕ := (jules_boxes * 150) / 100
def alice_boxes : ℕ := (john_boxes * 75) / 100
def gabriel_boxes : ℕ := (martin_boxes + alice_boxes) / 2

-- Theorem to prove
theorem gabriel_has_35_boxes : gabriel_boxes = 35 := by
  sorry

end gabriel_has_35_boxes_l4065_406507


namespace triangle_theorem_l4065_406514

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) :
  (t.c * Real.cos t.A + t.a * Real.cos t.C = 2 * t.b * Real.cos t.A) →
  Real.cos t.A = 1 / 2 ∧
  (t.a = Real.sqrt 7 ∧ t.b + t.c = 4) →
  (1 / 2 : ℝ) * t.b * t.c * Real.sqrt (1 - (1 / 2)^2) = 3 * Real.sqrt 3 / 4 :=
by sorry

-- Note: The area formula is expanded as (1/2) * b * c * sin A,
-- where sin A is written as sqrt(1 - cos^2 A)

end triangle_theorem_l4065_406514


namespace smallest_square_area_l4065_406567

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the diameter of the circle
def diameter : ℝ := 2 * radius

-- Define the side length of the square
def side_length : ℝ := diameter

-- Theorem: The area of the smallest square that can completely enclose a circle with a radius of 5 is 100
theorem smallest_square_area : side_length ^ 2 = 100 := by
  sorry

end smallest_square_area_l4065_406567


namespace smallest_divisible_by_million_l4065_406585

/-- A geometric sequence with first term a₁ and common ratio r -/
def geometric_sequence (a₁ : ℚ) (r : ℚ) : ℕ → ℚ :=
  λ n => a₁ * r^(n - 1)

/-- The nth term of the sequence is divisible by m -/
def is_divisible_by (seq : ℕ → ℚ) (n : ℕ) (m : ℕ) : Prop :=
  ∃ k : ℤ, seq n = k * (m : ℚ)

theorem smallest_divisible_by_million :
  let a₁ : ℚ := 3/4
  let a₂ : ℚ := 15
  let r : ℚ := a₂ / a₁
  let seq := geometric_sequence a₁ r
  (∀ n < 7, ¬ is_divisible_by seq n 1000000) ∧
  is_divisible_by seq 7 1000000 :=
by sorry

end smallest_divisible_by_million_l4065_406585


namespace max_value_under_constraints_l4065_406552

/-- Given real numbers x and y satisfying the given conditions, 
    the maximum value of 2x - y is 8. -/
theorem max_value_under_constraints (x y : ℝ) 
  (h1 : x + y - 7 ≤ 0) 
  (h2 : x - 3*y + 1 ≤ 0) 
  (h3 : 3*x - y - 5 ≥ 0) : 
  (∀ a b : ℝ, a + b - 7 ≤ 0 → a - 3*b + 1 ≤ 0 → 3*a - b - 5 ≥ 0 → 2*a - b ≤ 2*x - y) ∧ 
  2*x - y = 8 :=
sorry

end max_value_under_constraints_l4065_406552


namespace dvd_average_price_l4065_406562

theorem dvd_average_price (price1 price2 : ℚ) (count1 count2 : ℕ) 
  (h1 : price1 = 2)
  (h2 : price2 = 5)
  (h3 : count1 = 10)
  (h4 : count2 = 5) :
  (price1 * count1 + price2 * count2) / (count1 + count2) = 3 := by
sorry

end dvd_average_price_l4065_406562


namespace danny_watermelons_l4065_406597

theorem danny_watermelons (danny_slices_per_melon : ℕ) (sister_slices : ℕ) (total_slices : ℕ)
  (h1 : danny_slices_per_melon = 10)
  (h2 : sister_slices = 15)
  (h3 : total_slices = 45) :
  ∃ danny_melons : ℕ, danny_melons * danny_slices_per_melon + sister_slices = total_slices ∧ danny_melons = 3 := by
  sorry

end danny_watermelons_l4065_406597


namespace line_perpendicular_to_line_l4065_406535

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_line
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : subset_line_plane m β)
  (h3 : perpendicular_plane_plane α β) :
  perpendicular_line_line l m :=
sorry

end line_perpendicular_to_line_l4065_406535


namespace polynomial_simplification_l4065_406547

theorem polynomial_simplification (x : ℝ) :
  (2*x^6 + 3*x^5 + 4*x^4 + x^3 + x^2 + x + 20) - (x^6 + 4*x^5 + 2*x^4 - x^3 + 2*x^2 + x + 5) =
  x^6 - x^5 + 2*x^4 + 2*x^3 - x^2 + 15 := by
  sorry

end polynomial_simplification_l4065_406547


namespace select_books_count_l4065_406560

/-- The number of ways to select 5 books from 8 books, where 3 of the books form a trilogy that must be selected together. -/
def select_books : ℕ := 
  Nat.choose 5 2 + Nat.choose 5 5

/-- Theorem stating that the number of ways to select the books is 11. -/
theorem select_books_count : select_books = 11 := by
  sorry

end select_books_count_l4065_406560


namespace operation_problem_l4065_406526

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (diamond circ : Operation) :
  (apply_op diamond 10 4) / (apply_op circ 6 2) = 5 →
  (apply_op diamond 8 3) / (apply_op circ 10 5) = 8/5 := by
  sorry

end operation_problem_l4065_406526


namespace triangle_height_l4065_406586

/-- Given a triangle with area 615 m² and one side 123 meters, 
    the perpendicular height to that side is 10 meters. -/
theorem triangle_height (area : ℝ) (side : ℝ) (height : ℝ) : 
  area = 615 ∧ side = 123 → height = (2 * area) / side → height = 10 := by
  sorry

end triangle_height_l4065_406586


namespace envelope_area_l4065_406524

/-- The area of a rectangular envelope with width and length both equal to 4 inches is 16 square inches. -/
theorem envelope_area (width : ℝ) (length : ℝ) (h_width : width = 4) (h_length : length = 4) :
  width * length = 16 := by
  sorry

end envelope_area_l4065_406524


namespace product_97_103_l4065_406510

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end product_97_103_l4065_406510


namespace target_hit_probability_l4065_406542

def probability_hit : ℚ := 1 / 2

def total_shots : ℕ := 6

def successful_hits : ℕ := 3

def consecutive_hits : ℕ := 2

theorem target_hit_probability :
  (probability_hit ^ successful_hits) *
  ((1 - probability_hit) ^ (total_shots - successful_hits)) *
  (3 * (Nat.factorial 2 * Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2))) =
  3 / 16 := by
  sorry

end target_hit_probability_l4065_406542


namespace triangle_angles_l4065_406501

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a = 2 * b * Real.cos C ∧
  Real.sin A * Real.sin (B / 2 + C) = Real.sin C * (Real.sin (B / 2) + Real.sin A) →
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := by
sorry

end triangle_angles_l4065_406501


namespace a_income_l4065_406544

def income_ratio : ℚ := 5 / 4
def expenditure_ratio : ℚ := 3 / 2
def savings : ℕ := 1600

theorem a_income (a_income b_income a_expenditure b_expenditure : ℚ) 
  (h1 : a_income / b_income = income_ratio)
  (h2 : a_expenditure / b_expenditure = expenditure_ratio)
  (h3 : a_income - a_expenditure = savings)
  (h4 : b_income - b_expenditure = savings) :
  a_income = 4000 := by
  sorry

end a_income_l4065_406544


namespace quadratic_solution_sum_l4065_406553

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (4 * x^2 + 3 = 3 * x - 9) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 207/64 := by
sorry

end quadratic_solution_sum_l4065_406553


namespace bottle_production_l4065_406571

/-- Given that 5 identical machines produce 270 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2160 bottles in 4 minutes. -/
theorem bottle_production 
  (machines_5 : ℕ) 
  (bottles_per_minute_5 : ℕ) 
  (machines_10 : ℕ) 
  (minutes : ℕ) 
  (h1 : machines_5 = 5) 
  (h2 : bottles_per_minute_5 = 270) 
  (h3 : machines_10 = 10) 
  (h4 : minutes = 4) :
  (machines_10 * (bottles_per_minute_5 / machines_5) * minutes) = 2160 := by
  sorry

end bottle_production_l4065_406571


namespace jellybean_count_l4065_406516

theorem jellybean_count (remaining_ratio : ℝ) (days : ℕ) (final_count : ℕ) 
  (h1 : remaining_ratio = 0.75)
  (h2 : days = 3)
  (h3 : final_count = 27) :
  ∃ (original_count : ℕ), 
    (remaining_ratio ^ days) * (original_count : ℝ) = final_count ∧ 
    original_count = 64 := by
  sorry

end jellybean_count_l4065_406516


namespace A_intersect_B_eq_open_3_closed_2_l4065_406582

-- Define the sets A and B
def A : Set ℝ := {t : ℝ | ∀ x : ℝ, x^2 + 2*t*x - 4*t - 3 ≠ 0}
def B : Set ℝ := {t : ℝ | ∃ x : ℝ, x^2 + 2*t*x - 2*t = 0}

-- State the theorem
theorem A_intersect_B_eq_open_3_closed_2 : A ∩ B = Set.Ioc (-3) (-2) := by sorry

end A_intersect_B_eq_open_3_closed_2_l4065_406582


namespace inverse_existence_l4065_406596

-- Define the set of graph labels
inductive GraphLabel
  | A | B | C | D | E

-- Define a predicate for passing the horizontal line test
def passes_horizontal_line_test (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.A => False
  | GraphLabel.B => True
  | GraphLabel.C => True
  | GraphLabel.D => True
  | GraphLabel.E => False

-- Define a predicate for having an inverse
def has_inverse (g : GraphLabel) : Prop :=
  passes_horizontal_line_test g

-- Theorem statement
theorem inverse_existence (g : GraphLabel) :
  has_inverse g ↔ (g = GraphLabel.B ∨ g = GraphLabel.C ∨ g = GraphLabel.D) :=
by sorry

end inverse_existence_l4065_406596


namespace circle_area_diameter_13_l4065_406531

/-- The area of a circle with diameter 13 meters is π * (13/2)^2 square meters. -/
theorem circle_area_diameter_13 :
  let diameter : ℝ := 13
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = π * (13 / 2) ^ 2 := by sorry

end circle_area_diameter_13_l4065_406531


namespace proportion_equality_l4065_406538

theorem proportion_equality (x y : ℝ) (h1 : 3 * x = 2 * y) (h2 : y ≠ 0) : x / 2 = y / 3 := by
  sorry

end proportion_equality_l4065_406538


namespace circle_radius_tangent_to_lines_l4065_406550

/-- Theorem: For a circle with center (0, k) where k > 8, if the circle is tangent to the lines y = x, y = -x, and y = 8, then its radius is 8√2 + 8. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 8) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - k)^2 = (k - 8)^2}
  (∀ (x y : ℝ), (x = y ∧ (x, y) ∈ circle) → (x, y) ∈ {(x, y) | x = y}) →
  (∀ (x y : ℝ), (x = -y ∧ (x, y) ∈ circle) → (x, y) ∈ {(x, y) | x = -y}) →
  (∀ (x : ℝ), (x, 8) ∈ circle → x = 0) →
  k - 8 = 8 * (Real.sqrt 2 + 1) := by
sorry

end circle_radius_tangent_to_lines_l4065_406550


namespace line_intersection_theorem_l4065_406541

-- Define a type for lines
variable (Line : Type)

-- Define a property for line intersection
variable (intersect : Line → Line → Prop)

-- Define a property for lines passing through a common point
variable (pass_through_common_point : (Set Line) → Prop)

-- Define a property for lines lying in one plane
variable (lie_in_one_plane : (Set Line) → Prop)

-- The main theorem
theorem line_intersection_theorem (S : Set Line) :
  (∀ l1 l2 : Line, l1 ∈ S → l2 ∈ S → l1 ≠ l2 → intersect l1 l2) →
  (pass_through_common_point S ∨ lie_in_one_plane S) :=
by sorry


end line_intersection_theorem_l4065_406541


namespace cubic_equation_root_l4065_406592

theorem cubic_equation_root (a b : ℚ) : 
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = -1 - 3*Real.sqrt 2) →
  a = 19/17 := by
sorry

end cubic_equation_root_l4065_406592


namespace tens_digit_13_2023_l4065_406556

theorem tens_digit_13_2023 : ∃ n : ℕ, 13^2023 ≡ 90 + n [ZMOD 100] := by
  sorry

end tens_digit_13_2023_l4065_406556


namespace absolute_value_inequality_l4065_406540

theorem absolute_value_inequality (x : ℝ) : |5 - 2*x| ≥ 3 ↔ x < 1 ∨ x > 4 := by
  sorry

end absolute_value_inequality_l4065_406540


namespace sprite_volume_calculation_l4065_406565

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def total_cans : ℕ := 133

theorem sprite_volume_calculation :
  ∃ (can_volume sprite_volume : ℕ),
    can_volume > 0 ∧
    maaza_volume % can_volume = 0 ∧
    pepsi_volume % can_volume = 0 ∧
    sprite_volume % can_volume = 0 ∧
    maaza_volume / can_volume + pepsi_volume / can_volume + sprite_volume / can_volume = total_cans ∧
    sprite_volume = 368 := by
  sorry

end sprite_volume_calculation_l4065_406565


namespace parallel_lines_distance_l4065_406581

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 40, and 36, the distance between two adjacent parallel lines is 7.8. -/
theorem parallel_lines_distance (r : ℝ) : 
  let d := (4336 : ℝ) / 71
  (40 : ℝ) * r^2 = 16000 + 10 * d ∧ 
  (36 : ℝ) * r^2 = 11664 + 81 * d → 
  Real.sqrt d = 7.8 :=
by sorry

end parallel_lines_distance_l4065_406581


namespace range_of_k_value_of_k_l4065_406509

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2 - 2*k)*x + k^2 = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁ ≠ x₂

-- Define the additional condition
def root_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ 
    |x₁ + x₂| + 1 = x₁ * x₂

-- Theorem statements
theorem range_of_k (k : ℝ) : has_real_roots k → k ≤ 1/2 :=
sorry

theorem value_of_k : ∀ k : ℝ, has_real_roots k ∧ root_condition k → k = -3 :=
sorry

end range_of_k_value_of_k_l4065_406509


namespace minimum_N_l4065_406508

theorem minimum_N (k : ℕ) (h : k > 0) :
  let N := k * (2 * k^2 + 3 * k + 3)
  ∃ (S : Finset ℕ),
    (S.card = 2 * k + 1) ∧
    (∀ x ∈ S, x > 0) ∧
    (S.sum id > N) ∧
    (∀ T ⊆ S, T.card = k → T.sum id ≤ N / 2) ∧
    (∀ M < N, ¬∃ (S' : Finset ℕ),
      (S'.card = 2 * k + 1) ∧
      (∀ x ∈ S', x > 0) ∧
      (S'.sum id > M) ∧
      (∀ T ⊆ S', T.card = k → T.sum id ≤ M / 2)) :=
by sorry

end minimum_N_l4065_406508


namespace rs_length_l4065_406527

structure Tetrahedron where
  edges : Finset ℝ
  pq : ℝ

def valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges.card = 6 ∧ 
  t.pq ∈ t.edges ∧
  ∀ e ∈ t.edges, e > 0

theorem rs_length (t : Tetrahedron) 
  (h_valid : valid_tetrahedron t)
  (h_edges : t.edges = {9, 16, 22, 31, 39, 48})
  (h_pq : t.pq = 48) :
  ∃ rs ∈ t.edges, rs = 9 :=
sorry

end rs_length_l4065_406527


namespace article_cost_price_l4065_406513

theorem article_cost_price : ∃ C : ℝ, C = 400 ∧ 
  (1.05 * C - (0.95 * C + 0.1 * (0.95 * C)) = 2) := by
  sorry

end article_cost_price_l4065_406513


namespace discount_percentage_proof_l4065_406555

theorem discount_percentage_proof (pants_price : ℝ) (socks_price : ℝ) (total_after_discount : ℝ) :
  pants_price = 110 →
  socks_price = 60 →
  total_after_discount = 392 →
  let original_total := 4 * pants_price + 2 * socks_price
  let discount_amount := original_total - total_after_discount
  let discount_percentage := (discount_amount / original_total) * 100
  discount_percentage = 30 := by
  sorry

end discount_percentage_proof_l4065_406555


namespace yolas_past_weight_l4065_406570

/-- Yola's past weight given current weights and differences -/
theorem yolas_past_weight
  (yola_current : ℝ)
  (wanda_yola_current_diff : ℝ)
  (wanda_yola_past_diff : ℝ)
  (h1 : yola_current = 220)
  (h2 : wanda_yola_current_diff = 30)
  (h3 : wanda_yola_past_diff = 80) :
  yola_current - (wanda_yola_past_diff - wanda_yola_current_diff) = 170 :=
by
  sorry

end yolas_past_weight_l4065_406570


namespace sum_of_digits_6_11_l4065_406506

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_6_11 : 
  tens_digit (6^11) + ones_digit (6^11) = 13 := by sorry

end sum_of_digits_6_11_l4065_406506


namespace blue_parrots_count_l4065_406549

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) : 
  total = 108 →
  green_fraction = 5/6 →
  (total : ℚ) * (1 - green_fraction) = 18 := by
  sorry

end blue_parrots_count_l4065_406549


namespace target_line_is_correct_l4065_406595

/-- Given two lines in the xy-plane -/
def line1 : ℝ → ℝ → Prop := λ x y => x - y - 2 = 0
def line2 : ℝ → Prop := λ x => x - 2 = 0
def line3 : ℝ → ℝ → Prop := λ x y => x + y - 1 = 0

/-- The intersection point of line2 and line3 -/
def intersection_point : ℝ × ℝ := (2, -1)

/-- The equation of the line we want to prove -/
def target_line : ℝ → ℝ → Prop := λ x y => x - y - 3 = 0

/-- Main theorem -/
theorem target_line_is_correct : 
  (∀ x y, line1 x y ↔ ∃ k, target_line (x + k) (y + k)) ∧ 
  target_line intersection_point.1 intersection_point.2 :=
sorry

end target_line_is_correct_l4065_406595


namespace completing_square_transform_l4065_406559

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 - 2 = 0) := by
sorry

end completing_square_transform_l4065_406559


namespace power_of_1_01_gt_1000_l4065_406551

theorem power_of_1_01_gt_1000 : (1.01 : ℝ) ^ 1000 > 1000 := by
  sorry

end power_of_1_01_gt_1000_l4065_406551


namespace juniper_bones_l4065_406599

/-- Calculates the final number of bones Juniper has after transactions --/
def final_bones (initial : ℕ) : ℕ :=
  let additional := (initial * 50) / 100
  let total := initial + additional
  let stolen := (total * 25) / 100
  total - stolen

/-- Theorem stating that Juniper ends up with 5 bones --/
theorem juniper_bones : final_bones 4 = 5 := by
  sorry

end juniper_bones_l4065_406599


namespace equation_solution_l4065_406500

theorem equation_solution (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end equation_solution_l4065_406500


namespace number_equation_solution_l4065_406588

theorem number_equation_solution : ∃ x : ℝ, 33 + 3 * x = 48 ∧ x = 5 := by
  sorry

end number_equation_solution_l4065_406588


namespace tuesday_max_available_l4065_406502

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Dana
  | Evan

-- Define a function to represent availability
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => true
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => true
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Monday => true
  | Person.Dana, Day.Tuesday => true
  | Person.Dana, Day.Wednesday => false
  | Person.Dana, Day.Thursday => false
  | Person.Dana, Day.Friday => true
  | Person.Evan, Day.Monday => false
  | Person.Evan, Day.Tuesday => true
  | Person.Evan, Day.Wednesday => false
  | Person.Evan, Day.Thursday => true
  | Person.Evan, Day.Friday => true

-- Count available people for a given day
def countAvailable (d : Day) : Nat :=
  List.length (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Dana, Person.Evan])

-- Theorem: Tuesday has the maximum number of available people
theorem tuesday_max_available :
  ∀ d : Day, countAvailable Day.Tuesday ≥ countAvailable d :=
by
  sorry

end tuesday_max_available_l4065_406502


namespace union_of_A_and_B_l4065_406520

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_of_A_and_B : A ∪ B = {-3, -2, 2} := by sorry

end union_of_A_and_B_l4065_406520


namespace meaningful_expression_l4065_406573

/-- The expression sqrt(2-m) / (m+2) is meaningful if and only if m ≤ 2 and m ≠ -2 -/
theorem meaningful_expression (m : ℝ) : 
  (∃ x : ℝ, x^2 = 2 - m ∧ m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end meaningful_expression_l4065_406573


namespace valid_new_usage_exists_l4065_406578

/-- Represents the time spent on an app --/
structure AppTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the usage data for four apps --/
structure AppUsage where
  app1 : AppTime
  app2 : AppTime
  app3 : AppTime
  app4 : AppTime

/-- Checks if the new usage data is consistent with halving two app times --/
def is_valid_new_usage (old_usage new_usage : AppUsage) : Prop :=
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app3.time = old_usage.app3.time / 2 ∧
   new_usage.app2.time = old_usage.app2.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app2.time = old_usage.app2.time / 2 ∧
   new_usage.app3.time = old_usage.app3.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app2.time = old_usage.app2.time ∧ new_usage.app3.time = old_usage.app3.time) ∨
  (new_usage.app2.time = old_usage.app2.time / 2 ∧ new_usage.app3.time = old_usage.app3.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app2.time = old_usage.app2.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app3.time = old_usage.app3.time) ∨
  (new_usage.app3.time = old_usage.app3.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app2.time = old_usage.app2.time)

theorem valid_new_usage_exists (old_usage : AppUsage) :
  ∃ new_usage : AppUsage, is_valid_new_usage old_usage new_usage :=
sorry

end valid_new_usage_exists_l4065_406578


namespace olivia_won_five_games_l4065_406534

/-- Represents a contestant in the math quiz competition -/
inductive Contestant
| Liam
| Noah
| Olivia

/-- The number of games won by a contestant -/
def games_won (c : Contestant) : ℕ :=
  match c with
  | Contestant.Liam => 6
  | Contestant.Noah => 4
  | Contestant.Olivia => 5  -- This is what we want to prove

/-- The number of games lost by a contestant -/
def games_lost (c : Contestant) : ℕ :=
  match c with
  | Contestant.Liam => 3
  | Contestant.Noah => 4
  | Contestant.Olivia => 4

/-- The total number of games played by each contestant -/
def total_games (c : Contestant) : ℕ := games_won c + games_lost c

/-- Each win gives 1 point -/
def points (c : Contestant) : ℕ := games_won c

/-- Theorem stating that Olivia won 5 games -/
theorem olivia_won_five_games :
  (∀ c1 c2 : Contestant, c1 ≠ c2 → total_games c1 = total_games c2) →
  games_won Contestant.Olivia = 5 := by sorry

end olivia_won_five_games_l4065_406534


namespace perimeter_semicircular_pentagon_l4065_406519

/-- The perimeter of a region bounded by semicircular arcs constructed on each side of a regular pentagon --/
theorem perimeter_semicircular_pentagon (side_length : ℝ) : 
  side_length = 5 / π → 
  (5 : ℝ) * (π * side_length / 2) = 25 / 2 := by
  sorry

end perimeter_semicircular_pentagon_l4065_406519
