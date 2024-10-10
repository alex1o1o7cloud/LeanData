import Mathlib

namespace correct_sentence_structure_l715_71585

-- Define the possible options for filling the blanks
inductive BlankOption
  | indefiniteArticle : BlankOption
  | definiteArticle : BlankOption
  | empty : BlankOption

-- Define the sentence structure
structure Sentence where
  firstBlank : BlankOption
  secondBlank : BlankOption

-- Define the grammatical correctness of the sentence
def isGrammaticallyCorrect (s : Sentence) : Prop :=
  s.firstBlank = BlankOption.indefiniteArticle ∧ s.secondBlank = BlankOption.empty

-- Theorem: The sentence is grammatically correct when the first blank is "a" and the second is empty
theorem correct_sentence_structure :
  ∃ (s : Sentence), isGrammaticallyCorrect s :=
sorry

end correct_sentence_structure_l715_71585


namespace three_connected_iff_sequence_from_K4_l715_71548

/-- A simple graph. -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- A graph is 3-connected if removing any two vertices does not disconnect the graph. -/
def ThreeConnected (G : Graph V) : Prop := sorry

/-- The complete graph on 4 vertices. -/
def K4 (V : Type*) : Graph V := sorry

/-- Remove an edge from a graph. -/
def removeEdge (G : Graph V) (e : V × V) : Graph V := sorry

/-- Theorem 3.2.3 (Tutte, 1966): A graph is 3-connected if and only if it can be constructed
    from K4 by adding edges one at a time. -/
theorem three_connected_iff_sequence_from_K4 {V : Type*} (G : Graph V) :
  ThreeConnected G ↔ 
  ∃ (n : ℕ) (sequence : ℕ → Graph V),
    (sequence 0 = K4 V) ∧
    (sequence n = G) ∧
    (∀ i < n, ∃ e, sequence i = removeEdge (sequence (i + 1)) e) :=
  sorry

end three_connected_iff_sequence_from_K4_l715_71548


namespace intersection_A_B_l715_71528

def A : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

theorem intersection_A_B : A ∩ B = {x | -Real.pi < x ∧ x < 0 ∨ Real.pi < x ∧ x < 4} := by
  sorry

end intersection_A_B_l715_71528


namespace true_conjunction_with_negation_l715_71547

theorem true_conjunction_with_negation (p q : Prop) (hp : p) (hq : ¬q) : p ∧ ¬q := by
  sorry

end true_conjunction_with_negation_l715_71547


namespace opposite_of_one_over_twentythree_l715_71550

theorem opposite_of_one_over_twentythree :
  ∀ x : ℚ, x = 1 / 23 → -x = -(1 / 23) := by
  sorry

end opposite_of_one_over_twentythree_l715_71550


namespace inverse_composition_l715_71507

/-- A function g with specific values --/
def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 5
| 3 => 1
| 4 => 2
| 5 => 3

/-- The inverse of g --/
def g_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 4
| 3 => 5
| 4 => 1
| 5 => 2

/-- g is bijective --/
axiom g_bijective : Function.Bijective g

/-- g_inv is indeed the inverse of g --/
axiom g_inv_is_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

/-- The main theorem --/
theorem inverse_composition : g_inv (g_inv (g_inv 5)) = 1 := by sorry

end inverse_composition_l715_71507


namespace train_length_proof_l715_71526

/-- The length of two trains that pass each other under specific conditions -/
theorem train_length_proof (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 36) : 
  let L := (v_fast - v_slow) * t / (2 * 3600)
  L * 1000 = 50 := by sorry

end train_length_proof_l715_71526


namespace area_between_circles_l715_71559

theorem area_between_circles (r : ℝ) (R : ℝ) : 
  r = 3 →  -- radius of smaller circle
  R = 3 * r →  -- radius of larger circle is three times the smaller
  π * R^2 - π * r^2 = 72 * π := by
  sorry

end area_between_circles_l715_71559


namespace total_cookies_and_brownies_l715_71532

theorem total_cookies_and_brownies :
  let cookie_bags : ℕ := 272
  let cookies_per_bag : ℕ := 45
  let brownie_bags : ℕ := 158
  let brownies_per_bag : ℕ := 32
  cookie_bags * cookies_per_bag + brownie_bags * brownies_per_bag = 17296 := by
  sorry

end total_cookies_and_brownies_l715_71532


namespace probability_system_l715_71590

/-- Given a probability system with parameters p and q, prove that the probabilities x, y, and z satisfy specific relations. -/
theorem probability_system (p q x y z : ℝ) : 
  z = p * y + q * x → 
  x = p + q * x^2 → 
  y = q + p * y^2 → 
  x ≠ y → 
  p + q = 1 → 
  0 ≤ p ∧ p ≤ 1 → 
  0 ≤ q ∧ q ≤ 1 → 
  0 ≤ x ∧ x ≤ 1 → 
  0 ≤ y ∧ y ≤ 1 → 
  0 ≤ z ∧ z ≤ 1 → 
  x = 1 ∧ y = q / p ∧ z = 2 * q :=
by sorry

end probability_system_l715_71590


namespace opposite_numbers_quotient_l715_71541

theorem opposite_numbers_quotient (a b : ℝ) :
  a ≠ b → a = -b → a / b = -1 := by sorry

end opposite_numbers_quotient_l715_71541


namespace two_m_minus_b_is_zero_l715_71523

/-- A line passing through two points (1, 3) and (-1, 1) -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The line passes through the points (1, 3) and (-1, 1) -/
def line_through_points (l : Line) : Prop :=
  3 = l.m * 1 + l.b ∧ 1 = l.m * (-1) + l.b

/-- Theorem stating that 2m - b = 0 for the line passing through (1, 3) and (-1, 1) -/
theorem two_m_minus_b_is_zero (l : Line) (h : line_through_points l) : 
  2 * l.m - l.b = 0 := by
  sorry

end two_m_minus_b_is_zero_l715_71523


namespace product_of_solutions_l715_71527

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|20 / x₁ + 1| = 4) → 
  (|20 / x₂ + 1| = 4) → 
  (x₁ ≠ x₂) →
  (x₁ * x₂ = -80 / 3) :=
by sorry

end product_of_solutions_l715_71527


namespace tangent_circle_height_difference_l715_71536

/-- A circle tangent to the parabola y = x^2 + 1 at two points and inside the parabola -/
structure TangentCircle where
  /-- x-coordinate of one tangent point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_points : (a^2 + (a^2 + 1 - b)^2 = r^2) ∧ (a^2 + (a^2 + 1 - b)^2 = r^2)
  /-- The circle lies inside the parabola -/
  inside_parabola : ∀ x y, x^2 + (y - b)^2 = r^2 → y ≤ x^2 + 1

/-- The height difference between the center of the circle and the points of tangency is 1/2 -/
theorem tangent_circle_height_difference (c : TangentCircle) : 
  c.b - (c.a^2 + 1) = 1/2 := by
  sorry

end tangent_circle_height_difference_l715_71536


namespace f_of_2_equals_5_l715_71554

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_of_2_equals_5 : f 2 = 5 := by
  sorry

end f_of_2_equals_5_l715_71554


namespace three_from_nine_combination_l715_71592

theorem three_from_nine_combination : (Nat.choose 9 3) = 84 := by
  sorry

end three_from_nine_combination_l715_71592


namespace polar_midpoint_specific_l715_71571

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of the line segment with endpoints (10, π/4) and (10, 3π/4) in polar coordinates -/
theorem polar_midpoint_specific :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
sorry

end polar_midpoint_specific_l715_71571


namespace remaining_wire_length_l715_71530

-- Define the length of the iron wire
def wire_length (a b : ℝ) : ℝ := 5 * a + 4 * b

-- Define the perimeter of the rectangle
def rectangle_perimeter (a b : ℝ) : ℝ := 2 * (a + b)

-- Theorem statement
theorem remaining_wire_length (a b : ℝ) :
  wire_length a b - rectangle_perimeter a b = 3 * a + 2 * b := by
  sorry

end remaining_wire_length_l715_71530


namespace inequality_and_equality_condition_l715_71542

theorem inequality_and_equality_condition (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : 
  (2*x^2 - x + y + z)/(x + y^2 + z^2) + 
  (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
  (2*z^2 + x + y - z)/(x^2 + y^2 + z) ≥ 3 ∧
  ((2*x^2 - x + y + z)/(x + y^2 + z^2) + 
   (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
   (2*z^2 + x + y - z)/(x^2 + y^2 + z) = 3 ↔ x = y ∧ y = z) :=
by sorry

end inequality_and_equality_condition_l715_71542


namespace sum_of_four_digit_numbers_l715_71519

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum FourDigitNumber.value) = 399960 := by
  sorry

end sum_of_four_digit_numbers_l715_71519


namespace alcohol_dilution_l715_71583

theorem alcohol_dilution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hab : b < a) (hbc : c < b) :
  let initial_volume : ℝ := 1
  let first_dilution_volume : ℝ := a / b
  let second_dilution_volume : ℝ := a / (b + c)
  let total_water_used : ℝ := (first_dilution_volume - initial_volume) + (2 * second_dilution_volume - first_dilution_volume)
  total_water_used = 2 * a / (b + c) - 1 := by
sorry

end alcohol_dilution_l715_71583


namespace destiny_snack_bags_l715_71546

theorem destiny_snack_bags (chocolate_bars : Nat) (cookies : Nat) 
  (h1 : chocolate_bars = 18) (h2 : cookies = 12) :
  Nat.gcd chocolate_bars cookies = 6 := by
  sorry

end destiny_snack_bags_l715_71546


namespace problem_proof_l715_71562

theorem problem_proof : |Real.sqrt 3 - 2| - Real.sqrt ((-3)^2) + 2 * Real.sqrt 9 = 5 - Real.sqrt 3 := by
  sorry

end problem_proof_l715_71562


namespace profit_distribution_correct_l715_71565

def total_profit : ℕ := 280000

def shekhar_percentage : ℚ := 28 / 100
def rajeev_percentage : ℚ := 22 / 100
def jatin_percentage : ℚ := 20 / 100
def simran_percentage : ℚ := 18 / 100
def ramesh_percentage : ℚ := 12 / 100

def shekhar_share : ℕ := (shekhar_percentage * total_profit).num.toNat
def rajeev_share : ℕ := (rajeev_percentage * total_profit).num.toNat
def jatin_share : ℕ := (jatin_percentage * total_profit).num.toNat
def simran_share : ℕ := (simran_percentage * total_profit).num.toNat
def ramesh_share : ℕ := (ramesh_percentage * total_profit).num.toNat

theorem profit_distribution_correct :
  shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit :=
by sorry

end profit_distribution_correct_l715_71565


namespace negative_majority_sequence_l715_71591

theorem negative_majority_sequence :
  ∃ (x : Fin 2004 → ℤ),
    (∀ k : Fin 2001, x (k + 3) = x (k + 2) + x k * x (k + 1)) ∧
    (∃ n : ℕ, 2 * n > 2004 ∧ (∃ S : Finset (Fin 2004), S.card = n ∧ ∀ i ∈ S, x i < 0)) := by
  sorry

end negative_majority_sequence_l715_71591


namespace ellipse_focal_chord_area_l715_71570

/-- Given an ellipse with equation x²/4 + y²/m = 1 (m > 0), where the focal chord F₁F₂ is the diameter
    of a circle intersecting the ellipse at point P in the first quadrant, if the area of triangle PF₁F₂
    is 1, then m = 1. -/
theorem ellipse_focal_chord_area (m : ℝ) (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  m > 0 → 
  x^2 / 4 + y^2 / m = 1 →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 16 →  -- F₁F₂ is diameter of circle with radius 2
  P.1^2 / 4 + P.2^2 / m = 1 →  -- P is on the ellipse
  P.1 ≥ 0 ∧ P.2 ≥ 0 →  -- P is in the first quadrant
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 16 →  -- P is on the circle
  abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 1 →  -- Area of triangle PF₁F₂ is 1
  m = 1 := by
sorry

end ellipse_focal_chord_area_l715_71570


namespace arithmetic_sequence_minimum_l715_71515

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive terms
  (∀ k, a (k + 1) = 2 * a k) →  -- Common ratio q = 2
  (Real.sqrt (a m * a n) = 4 * a 1) →  -- Condition on a_m and a_n
  (∃ p q : ℕ, 1/p + 4/q ≤ 1/m + 4/n) →  -- Existence of minimum
  1/m + 4/n ≥ 3/2 :=
by sorry

end arithmetic_sequence_minimum_l715_71515


namespace gcd_multiple_relation_l715_71553

theorem gcd_multiple_relation (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = 7 * b) :
  Nat.gcd a b = b :=
by sorry

end gcd_multiple_relation_l715_71553


namespace equality_condition_l715_71594

theorem equality_condition (a b c : ℝ) : 
  a = b + c + 2 → (a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1) := by
  sorry

end equality_condition_l715_71594


namespace erica_safari_lions_erica_saw_three_lions_l715_71552

/-- Prove that Erica saw 3 lions on Saturday during her safari -/
theorem erica_safari_lions : ℕ → Prop := fun n =>
  let total_animals : ℕ := 20
  let saturday_elephants : ℕ := 2
  let sunday_animals : ℕ := 2 + 5  -- 2 buffaloes and 5 leopards
  let monday_animals : ℕ := 5 + 3  -- 5 rhinos and 3 warthogs
  n = total_animals - (saturday_elephants + sunday_animals + monday_animals)

/-- The number of lions Erica saw on Saturday is 3 -/
theorem erica_saw_three_lions : erica_safari_lions 3 := by
  sorry

end erica_safari_lions_erica_saw_three_lions_l715_71552


namespace sum_less_than_one_l715_71549

theorem sum_less_than_one (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) : 
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := by
  sorry

end sum_less_than_one_l715_71549


namespace two_digit_number_interchange_l715_71560

theorem two_digit_number_interchange (a b j : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 10 * a + b = j * (a + b)) :
  10 * b + a = (10 * j - 9) * (a + b) :=
sorry

end two_digit_number_interchange_l715_71560


namespace tan_sum_one_fortyfour_l715_71588

theorem tan_sum_one_fortyfour : (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2 := by
  sorry

end tan_sum_one_fortyfour_l715_71588


namespace log_two_plus_log_five_equals_one_l715_71555

theorem log_two_plus_log_five_equals_one : Real.log 2 + Real.log 5 = 1 := by
  sorry

end log_two_plus_log_five_equals_one_l715_71555


namespace selling_price_l715_71551

/-- Given an original price and a percentage increase, calculate the selling price -/
theorem selling_price (a : ℝ) : (a * (1 + 0.1)) = 1.1 * a := by sorry

end selling_price_l715_71551


namespace max_friends_theorem_l715_71509

/-- Represents the configuration of gnomes in towers --/
structure GnomeCity (n : ℕ) where
  (n_even : Even n)
  (n_pos : 0 < n)

/-- The maximal number of pairs of gnomes which are friends --/
def max_friends (city : GnomeCity n) : ℕ := n^3 / 4

/-- Theorem stating the maximal number of pairs of gnomes which are friends --/
theorem max_friends_theorem (n : ℕ) (city : GnomeCity n) :
  max_friends city = n^3 / 4 := by sorry

end max_friends_theorem_l715_71509


namespace min_a2_plus_b2_l715_71598

-- Define the quadratic function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (2*b + 1) * x - a - 2

-- State the theorem
theorem min_a2_plus_b2 (a b : ℝ) (ha : a ≠ 0) 
  (hroot : ∃ x ∈ Set.Icc 3 4, f a b x = 0) : 
  ∃ min_val : ℝ, (∀ a' b' : ℝ, a'^2 + b'^2 ≥ min_val) ∧ 
  (∃ a₀ b₀ : ℝ, a₀^2 + b₀^2 = min_val) ∧ min_val = 1/100 :=
sorry

end min_a2_plus_b2_l715_71598


namespace company_females_count_l715_71537

theorem company_females_count (total_employees : ℕ) 
  (advanced_degrees : ℕ) (males_college_only : ℕ) (females_advanced : ℕ) 
  (h1 : total_employees = 148)
  (h2 : advanced_degrees = 78)
  (h3 : males_college_only = 31)
  (h4 : females_advanced = 53) :
  total_employees - advanced_degrees - males_college_only + females_advanced = 92 :=
by sorry

end company_females_count_l715_71537


namespace negation_of_proposition_quadratic_inequality_negation_l715_71563

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬∀ x, p x) ↔ ∃ x, ¬(p x) := by sorry

theorem quadratic_inequality_negation :
  (¬∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := by sorry

end negation_of_proposition_quadratic_inequality_negation_l715_71563


namespace power_of_81_l715_71502

theorem power_of_81 : (81 : ℝ) ^ (5/2) = 59049 := by
  sorry

end power_of_81_l715_71502


namespace min_ratio_four_digit_number_l715_71561

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The theorem stating that 1099 minimizes x/y for four-digit numbers -/
theorem min_ratio_four_digit_number :
  ∀ (x : FourDigitNumber),
    (x.val : ℚ) / digit_sum x.val ≥ 1099 / digit_sum 1099 := by sorry

end min_ratio_four_digit_number_l715_71561


namespace fruit_salad_count_l715_71524

/-- Given a fruit salad with red grapes, green grapes, and raspberries, 
    this theorem proves the total number of fruits in the salad. -/
theorem fruit_salad_count (red_grapes green_grapes raspberries : ℕ) : 
  red_grapes = 67 →
  red_grapes = 3 * green_grapes + 7 →
  raspberries = green_grapes - 5 →
  red_grapes + green_grapes + raspberries = 102 := by
  sorry

#check fruit_salad_count

end fruit_salad_count_l715_71524


namespace patricia_books_l715_71516

def book_tournament (candice amanda kara patricia : ℕ) : Prop :=
  candice = 3 * amanda ∧
  kara = amanda / 2 ∧
  patricia = 7 * kara ∧
  candice = 18

theorem patricia_books :
  ∀ candice amanda kara patricia : ℕ,
    book_tournament candice amanda kara patricia →
    patricia = 21 :=
by
  sorry

end patricia_books_l715_71516


namespace ball_distribution_theorem_l715_71586

theorem ball_distribution_theorem (num_colors num_students total_balls min_balls_per_color min_balls_per_box : ℕ) 
  (h1 : num_colors = 20)
  (h2 : num_students = 20)
  (h3 : total_balls = 800)
  (h4 : min_balls_per_color ≥ 10)
  (h5 : min_balls_per_box ≥ 10) :
  ∃ (balls_per_student : ℕ), 
    balls_per_student * num_students = total_balls ∧ 
    ∃ (num_boxes : ℕ), 
      num_boxes % num_students = 0 ∧ 
      num_boxes * min_balls_per_box ≤ total_balls ∧
      (num_boxes / num_students) * min_balls_per_box = balls_per_student :=
by
  sorry

end ball_distribution_theorem_l715_71586


namespace circle_no_intersection_with_axes_l715_71568

theorem circle_no_intersection_with_axes (k : ℝ) :
  (k > 0) →
  (∀ x y : ℝ, x^2 + y^2 - 2*k*x + 2*y + 2 = 0 → (x ≠ 0 ∧ y ≠ 0)) →
  k > 1 ∧ k < Real.sqrt 2 := by
  sorry

end circle_no_intersection_with_axes_l715_71568


namespace angle_y_is_90_l715_71501

-- Define the angles
def angle_ABC : ℝ := 120
def angle_ABE : ℝ := 30

-- Define the theorem
theorem angle_y_is_90 :
  ∀ (angle_y angle_ABD : ℝ),
  -- Condition 3
  angle_ABD + angle_ABC = 180 →
  -- Condition 4
  angle_ABE + angle_y = 180 →
  -- Condition 5 (using angle_y instead of explicitly stating the third angle)
  angle_y + angle_ABD + angle_ABE = 180 →
  -- Conclusion
  angle_y = 90 := by
sorry

end angle_y_is_90_l715_71501


namespace common_divisors_9240_8820_l715_71503

theorem common_divisors_9240_8820 : 
  (Nat.divisors (Nat.gcd 9240 8820)).card = 24 := by sorry

end common_divisors_9240_8820_l715_71503


namespace stimulus_check_distribution_l715_71533

theorem stimulus_check_distribution (total amount_to_wife amount_to_first_son amount_to_second_son savings : ℚ) :
  total = 2000 ∧
  amount_to_wife = (2 / 5) * total ∧
  amount_to_first_son = (2 / 5) * (total - amount_to_wife) ∧
  savings = 432 ∧
  amount_to_second_son = total - amount_to_wife - amount_to_first_son - savings →
  amount_to_second_son / (total - amount_to_wife - amount_to_first_son) = 2 / 5 := by
  sorry

end stimulus_check_distribution_l715_71533


namespace complex_magnitude_product_l715_71514

theorem complex_magnitude_product : 3 * Complex.abs (1 - 3*I) * Complex.abs (1 + 3*I) = 30 := by
  sorry

end complex_magnitude_product_l715_71514


namespace stock_value_comparison_l715_71506

def initial_investment : ℝ := 200

def first_year_change_DD : ℝ := 1.10
def first_year_change_EE : ℝ := 0.85
def first_year_change_FF : ℝ := 1.05

def second_year_change_DD : ℝ := 1.05
def second_year_change_EE : ℝ := 1.15
def second_year_change_FF : ℝ := 0.90

def D : ℝ := initial_investment * first_year_change_DD * second_year_change_DD
def E : ℝ := initial_investment * first_year_change_EE * second_year_change_EE
def F : ℝ := initial_investment * first_year_change_FF * second_year_change_FF

theorem stock_value_comparison : F < E ∧ E < D := by
  sorry

end stock_value_comparison_l715_71506


namespace domain_of_y_l715_71595

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function y
def y (f : Set ℝ) (x : ℝ) : Prop :=
  (x + 3 ∈ f) ∧ (x^2 ∈ f)

-- Theorem statement
theorem domain_of_y (f : Set ℝ) :
  f = Set.Icc 0 4 →
  {x : ℝ | y f x} = Set.Icc (-2) 1 := by
sorry

end domain_of_y_l715_71595


namespace missing_carton_dimension_l715_71518

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℝ :=
  dims.length * dims.width * dims.height

/-- Theorem: Given the dimensions of a carton and soap box, if 300 soap boxes fit exactly in the carton,
    then the missing dimension of the carton is 25 inches -/
theorem missing_carton_dimension
  (x : ℝ)
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton = { length := x, width := 48, height := 60 })
  (h2 : soap = { length := 8, width := 6, height := 5 })
  (h3 : (boxVolume carton) / (boxVolume soap) = 300)
  : x = 25 := by
  sorry

#check missing_carton_dimension

end missing_carton_dimension_l715_71518


namespace no_positive_and_negative_rational_l715_71538

theorem no_positive_and_negative_rational : ¬∃ (q : ℚ), q > 0 ∧ q < 0 := by
  sorry

end no_positive_and_negative_rational_l715_71538


namespace chess_tournament_solution_l715_71511

-- Define the tournament structure
structure ChessTournament where
  grade8_students : ℕ
  grade7_points : ℕ
  grade8_points : ℕ

-- Define the tournament conditions
def valid_tournament (t : ChessTournament) : Prop :=
  t.grade7_points = 8 ∧
  t.grade8_points * t.grade8_students = (t.grade8_students + 2) * (t.grade8_students + 1) / 2 - 8

-- Theorem statement
theorem chess_tournament_solution (t : ChessTournament) :
  valid_tournament t → (t.grade8_students = 7 ∨ t.grade8_students = 14) :=
by
  sorry

end chess_tournament_solution_l715_71511


namespace intersection_points_count_l715_71556

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def intersectsCircle (l : Line) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def intersectLines (l1 l2 : Line) : Prop :=
  sorry

/-- Function to count intersection points between a line and a circle -/
def countIntersections (l : Line) (c : Circle) : ℕ :=
  sorry

/-- Main theorem -/
theorem intersection_points_count 
  (c : Circle) (l1 l2 : Line) 
  (h1 : isTangent l1 c)
  (h2 : intersectsCircle l2 c)
  (h3 : ¬ isTangent l2 c)
  (h4 : intersectLines l1 l2) :
  countIntersections l1 c + countIntersections l2 c = 3 :=
sorry

end intersection_points_count_l715_71556


namespace no_two_digit_divisible_by_reverse_l715_71535

theorem no_two_digit_divisible_by_reverse : ¬ ∃ (a b : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ a ≠ b ∧ 
  ∃ (k : ℕ), k > 1 ∧ (10 * a + b) = k * (10 * b + a) :=
by sorry

end no_two_digit_divisible_by_reverse_l715_71535


namespace pelican_migration_l715_71539

/-- Represents the number of Pelicans originally in Shark Bite Cove -/
def original_pelicans : ℕ := 30

/-- Represents the number of sharks in Pelican Bay -/
def sharks_in_pelican_bay : ℕ := 60

/-- Represents the number of Pelicans remaining in Shark Bite Cove -/
def remaining_pelicans : ℕ := 20

/-- The fraction of Pelicans that moved from Shark Bite Cove to Pelican Bay -/
def fraction_moved : ℚ := 1 / 3

theorem pelican_migration :
  (sharks_in_pelican_bay = 2 * original_pelicans) ∧
  (remaining_pelicans < original_pelicans) ∧
  (fraction_moved = (original_pelicans - remaining_pelicans : ℚ) / original_pelicans) :=
by sorry

end pelican_migration_l715_71539


namespace sum_of_even_integers_302_to_400_l715_71510

theorem sum_of_even_integers_302_to_400 (sum_first_50 : ℕ) (sum_302_to_400 : ℕ) : 
  sum_first_50 = 2550 → sum_302_to_400 = 17550 → sum_302_to_400 - sum_first_50 = 15000 := by
  sorry

end sum_of_even_integers_302_to_400_l715_71510


namespace matts_assignment_problems_l715_71589

/-- The number of minutes it takes Matt to solve one problem with a calculator -/
def time_with_calculator : ℕ := 2

/-- The number of minutes it takes Matt to solve one problem without a calculator -/
def time_without_calculator : ℕ := 5

/-- The total number of minutes saved by using a calculator -/
def time_saved : ℕ := 60

/-- The number of problems in Matt's assignment -/
def number_of_problems : ℕ := 20

theorem matts_assignment_problems :
  (time_without_calculator - time_with_calculator) * number_of_problems = time_saved :=
by sorry

end matts_assignment_problems_l715_71589


namespace lemonade_sales_difference_l715_71545

/-- 
Given:
- x: number of glasses of plain lemonade sold
- y: number of glasses of strawberry lemonade sold
- p: price of each glass of plain lemonade
- s: price of each glass of strawberry lemonade
- The total amount from plain lemonade is 1.5 times the total amount from strawberry lemonade

Prove that the difference between the total amount made from plain lemonade and 
strawberry lemonade is equal to 0.5 * (y * s)
-/
theorem lemonade_sales_difference 
  (x y p s : ℝ) 
  (h : x * p = 1.5 * (y * s)) : 
  x * p - y * s = 0.5 * (y * s) := by
  sorry


end lemonade_sales_difference_l715_71545


namespace price_difference_is_500_l715_71544

/-- The price difference between enhanced and basic computers -/
def price_difference (total_basic_printer : ℝ) (price_basic : ℝ) : ℝ :=
  let price_printer := total_basic_printer - price_basic
  let price_enhanced := 8 * price_printer - price_printer
  price_enhanced - price_basic

/-- Theorem stating the price difference between enhanced and basic computers -/
theorem price_difference_is_500 :
  price_difference 2500 2125 = 500 := by
  sorry

end price_difference_is_500_l715_71544


namespace executive_committee_formation_l715_71578

theorem executive_committee_formation (total_members : ℕ) (committee_size : ℕ) (president : ℕ) :
  total_members = 30 →
  committee_size = 5 →
  president = 1 →
  Nat.choose (total_members - president) (committee_size - president) = 25839 :=
by sorry

end executive_committee_formation_l715_71578


namespace no_solution_implies_a_equals_six_l715_71573

theorem no_solution_implies_a_equals_six (a : ℝ) : 
  (∀ x y : ℝ, (x + 2*y = 4 ∧ 3*x + a*y = 6) → False) → a = 6 :=
by sorry

end no_solution_implies_a_equals_six_l715_71573


namespace red_bead_count_l715_71580

/-- Represents a necklace with blue and red beads. -/
structure Necklace where
  blue_count : Nat
  red_count : Nat
  is_valid : Bool

/-- Checks if a necklace satisfies the given conditions. -/
def is_valid_necklace (n : Necklace) : Prop :=
  n.blue_count = 30 ∧
  n.is_valid = true ∧
  n.red_count > 0 ∧
  n.red_count % 2 = 0 ∧
  n.red_count = 2 * n.blue_count

theorem red_bead_count (n : Necklace) :
  is_valid_necklace n → n.red_count = 60 := by
  sorry

#check red_bead_count

end red_bead_count_l715_71580


namespace line_x_intercept_l715_71566

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

/-- The theorem stating that the line through (6, 2) and (2, 6) has x-intercept at x = 8 -/
theorem line_x_intercept :
  let l : Line := { x₁ := 6, y₁ := 2, x₂ := 2, y₂ := 6 }
  xIntercept l = 8 := by sorry

end line_x_intercept_l715_71566


namespace store_transaction_result_l715_71579

theorem store_transaction_result : 
  let selling_price : ℝ := 960
  let profit_margin : ℝ := 0.2
  let cost_profit_item : ℝ := selling_price / (1 + profit_margin)
  let cost_loss_item : ℝ := selling_price / (1 - profit_margin)
  let total_cost : ℝ := cost_profit_item + cost_loss_item
  let total_revenue : ℝ := 2 * selling_price
  total_cost - total_revenue = 80
  := by sorry

end store_transaction_result_l715_71579


namespace f_of_three_equals_e_squared_l715_71582

theorem f_of_three_equals_e_squared 
  (f : ℝ → ℝ) 
  (h : ∀ x > 0, f (Real.log x + 1) = x) : 
  f 3 = Real.exp 2 := by
  sorry

end f_of_three_equals_e_squared_l715_71582


namespace law_of_sines_symmetry_l715_71569

/-- The Law of Sines for a triangle ABC with sides a, b, c and angles A, B, C -/
def law_of_sines (a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- A property representing symmetry in mathematical expressions -/
def has_symmetry (P : Prop) : Prop :=
  -- This is a placeholder definition. In a real implementation, 
  -- this would need to be defined based on specific criteria for symmetry.
  true

/-- Theorem stating that the Law of Sines exhibits mathematical symmetry -/
theorem law_of_sines_symmetry (a b c : ℝ) (A B C : ℝ) :
  has_symmetry (law_of_sines a b c A B C) :=
sorry

end law_of_sines_symmetry_l715_71569


namespace x_value_l715_71587

theorem x_value (x y : ℚ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := by
  sorry

end x_value_l715_71587


namespace multiplication_of_powers_l715_71543

theorem multiplication_of_powers (b : ℝ) : 3 * b^3 * (2 * b^2) = 6 * b^5 := by
  sorry

end multiplication_of_powers_l715_71543


namespace value_calculation_l715_71572

theorem value_calculation (N : ℝ) (h : 0.4 * N = 300) : (1/4) * (1/3) * (2/5) * N = 25 := by
  sorry

end value_calculation_l715_71572


namespace remaining_distance_l715_71597

def total_journey : ℕ := 1200
def distance_driven : ℕ := 642

theorem remaining_distance : total_journey - distance_driven = 558 := by
  sorry

end remaining_distance_l715_71597


namespace geometric_sequence_incorrect_statement_l715_71577

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_incorrect_statement
  (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_sequence a q) (h2 : q ≠ 1) :
  ¬(a 2 > a 1 → ∀ n : ℕ, a (n + 1) > a n) :=
sorry

end geometric_sequence_incorrect_statement_l715_71577


namespace coefficients_of_given_equation_l715_71564

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation 2x^2 + x - 5 = 0 -/
def givenEquation : QuadraticEquation := ⟨2, 1, -5⟩

theorem coefficients_of_given_equation :
  givenEquation.a = 2 ∧ givenEquation.b = 1 ∧ givenEquation.c = -5 := by
  sorry

end coefficients_of_given_equation_l715_71564


namespace commission_change_point_l715_71567

/-- The sales amount where the commission rate changes -/
def X : ℝ := 10000

/-- The total sales amount -/
def total_sales : ℝ := 32500

/-- The amount remitted to the parent company -/
def remitted_amount : ℝ := 31100

/-- The commission rate for sales up to X -/
def commission_rate_low : ℝ := 0.05

/-- The commission rate for sales exceeding X -/
def commission_rate_high : ℝ := 0.04

theorem commission_change_point :
  X = 10000 ∧
  total_sales = 32500 ∧
  remitted_amount = 31100 ∧
  commission_rate_low = 0.05 ∧
  commission_rate_high = 0.04 ∧
  remitted_amount = total_sales - (commission_rate_low * X + commission_rate_high * (total_sales - X)) :=
by sorry

end commission_change_point_l715_71567


namespace shifted_quadratic_function_l715_71584

/-- A quadratic function -/
def f (x : ℝ) : ℝ := -x^2

/-- Horizontal shift of a function -/
def horizontalShift (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x + h)

/-- Vertical shift of a function -/
def verticalShift (f : ℝ → ℝ) (v : ℝ) : ℝ → ℝ := fun x ↦ f x + v

/-- The shifted function -/
def g : ℝ → ℝ := verticalShift (horizontalShift f 1) 3

theorem shifted_quadratic_function :
  ∀ x : ℝ, g x = -(x + 1)^2 + 3 :=
by sorry

end shifted_quadratic_function_l715_71584


namespace cone_lateral_surface_area_l715_71529

/-- The lateral surface area of a cone with base radius 3 and central angle of its lateral surface unfolded diagram 90° is 36π. -/
theorem cone_lateral_surface_area :
  let base_radius : ℝ := 3
  let central_angle : ℝ := 90
  let lateral_surface_area : ℝ := (1 / 2) * (2 * Real.pi * base_radius) * (4 * base_radius)
  lateral_surface_area = 36 * Real.pi :=
by sorry

end cone_lateral_surface_area_l715_71529


namespace cos_equality_317_degrees_l715_71599

theorem cos_equality_317_degrees (n : ℕ) (h1 : n ≤ 180) (h2 : Real.cos (n * π / 180) = Real.cos (317 * π / 180)) : n = 43 := by
  sorry

end cos_equality_317_degrees_l715_71599


namespace cube_cutting_l715_71520

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
sorry

end cube_cutting_l715_71520


namespace multiply_negatives_l715_71576

theorem multiply_negatives : (-4 : ℚ) * (-(-(1/2))) = -2 := by
  sorry

end multiply_negatives_l715_71576


namespace arrangement_count_is_72_l715_71521

/-- Represents the number of ways to arrange 5 people with specific conditions -/
def arrangement_count : ℕ := 72

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of ways to arrange C and D together -/
def cd_arrangements : ℕ := 2

/-- The number of ways to arrange 3 entities (C-D unit, another person, and a space) -/
def entity_arrangements : ℕ := 6

/-- The number of ways to place A and B not adjacent in the remaining spaces -/
def ab_placements : ℕ := 6

/-- Theorem stating that the number of arrangements satisfying the conditions is 72 -/
theorem arrangement_count_is_72 :
  arrangement_count = cd_arrangements * entity_arrangements * ab_placements :=
sorry

end arrangement_count_is_72_l715_71521


namespace bubble_sort_probability_correct_l715_71593

/-- The probability of the 25th element in a random permutation of 50 distinct real numbers
    ending up in the 40th position after one bubble pass -/
def bubble_sort_probability : ℚ :=
  1 / 1640

/-- The sequence length -/
def n : ℕ := 50

/-- The initial position of the element we're tracking -/
def initial_position : ℕ := 25

/-- The final position of the element we're tracking -/
def final_position : ℕ := 40

theorem bubble_sort_probability_correct :
  bubble_sort_probability = 1 / 1640 ∧ n = 50 ∧ initial_position = 25 ∧ final_position = 40 := by
  sorry

end bubble_sort_probability_correct_l715_71593


namespace product_of_x_values_l715_71525

theorem product_of_x_values (x : ℝ) : 
  (|10 / x - 4| = 3) → 
  (∃ y : ℝ, (|10 / y - 4| = 3) ∧ x * y = 100 / 7) :=
by sorry

end product_of_x_values_l715_71525


namespace arithmetic_mean_of_fractions_l715_71534

theorem arithmetic_mean_of_fractions :
  (5 : ℚ) / 6 = ((3 : ℚ) / 4 + (7 : ℚ) / 8) / 2 := by sorry

end arithmetic_mean_of_fractions_l715_71534


namespace triangle_area_is_24_l715_71531

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (0, 6)
def vertex3 : ℝ × ℝ := (8, 10)

-- Define the triangle area calculation function
def triangleArea (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- State the theorem
theorem triangle_area_is_24 :
  triangleArea vertex1 vertex2 vertex3 = 24 := by
  sorry


end triangle_area_is_24_l715_71531


namespace set_union_implies_a_value_l715_71508

theorem set_union_implies_a_value (a : ℝ) :
  let A : Set ℝ := {2^a, 3}
  let B : Set ℝ := {2, 3}
  A ∪ B = {2, 3, 4} →
  a = 2 := by
sorry

end set_union_implies_a_value_l715_71508


namespace potato_fetch_time_l715_71574

-- Define the constants
def football_fields : ℕ := 6
def yards_per_field : ℕ := 200
def feet_per_yard : ℕ := 3
def dog_speed : ℕ := 400  -- feet per minute

-- Define the theorem
theorem potato_fetch_time :
  let total_distance : ℕ := football_fields * yards_per_field * feet_per_yard
  let fetch_time : ℕ := total_distance / dog_speed
  fetch_time = 9 := by sorry

end potato_fetch_time_l715_71574


namespace num_tough_weeks_is_three_l715_71512

def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def num_good_weeks : ℕ := 5
def total_sales : ℕ := 10400

theorem num_tough_weeks_is_three :
  ∃ (num_tough_weeks : ℕ),
    num_tough_weeks * tough_week_sales + num_good_weeks * good_week_sales = total_sales ∧
    num_tough_weeks = 3 :=
by sorry

end num_tough_weeks_is_three_l715_71512


namespace right_quadrilateral_area_l715_71596

/-- A quadrilateral with right angles at B and D, diagonal AC = 3, and two sides with distinct integer lengths. -/
structure RightQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  right_angle_B : AB * BC = 0
  right_angle_D : CD * DA = 0
  diagonal_AC : AB^2 + BC^2 = 9
  distinct_integer_sides : ∃ (x y : ℕ), x ≠ y ∧ ((AB = x ∧ CD = y) ∨ (AB = x ∧ DA = y) ∨ (BC = x ∧ CD = y) ∨ (BC = x ∧ DA = y))

/-- The area of a RightQuadrilateral is √2 + √5. -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : Real.sqrt 2 + Real.sqrt 5 = q.AB * q.BC / 2 + q.CD * q.DA / 2 := by
  sorry

end right_quadrilateral_area_l715_71596


namespace existence_of_non_square_product_l715_71558

theorem existence_of_non_square_product (d : ℕ) 
  (h_d_pos : d > 0) 
  (h_d_neq_2 : d ≠ 2) 
  (h_d_neq_5 : d ≠ 5) 
  (h_d_neq_13 : d ≠ 13) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               a ≠ b ∧ 
               ¬∃ (k : ℕ), a * b - 1 = k * k :=
by
  sorry

end existence_of_non_square_product_l715_71558


namespace problem_statement_l715_71581

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 1| + |x + a|

-- Define the theorem
theorem problem_statement 
  (a b : ℝ)
  (m n : ℝ)
  (h1 : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1)
  (h2 : m > 0)
  (h3 : n > 0)
  (h4 : 1/(2*m) + 2/n + 2*a = 0) :
  a = -1 ∧ 4*m^2 + n^2 ≥ 4 := by
  sorry

end problem_statement_l715_71581


namespace sample_size_is_30_l715_71575

/-- Represents the company's employee data and sampling information -/
structure CompanyData where
  total_employees : ℕ
  young_employees : ℕ
  sample_young : ℕ
  h_young_le_total : young_employees ≤ total_employees

/-- Calculates the sample size based on stratified sampling -/
def calculate_sample_size (data : CompanyData) : ℕ :=
  (data.sample_young * data.total_employees) / data.young_employees

/-- Proves that the sample size is 30 given the specific company data -/
theorem sample_size_is_30 (data : CompanyData) 
  (h_total : data.total_employees = 900)
  (h_young : data.young_employees = 450)
  (h_sample_young : data.sample_young = 15) :
  calculate_sample_size data = 30 := by
  sorry

#eval calculate_sample_size ⟨900, 450, 15, by norm_num⟩

end sample_size_is_30_l715_71575


namespace fraction_simplification_l715_71522

theorem fraction_simplification (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by
  sorry

end fraction_simplification_l715_71522


namespace midpoint_coordinate_sum_l715_71513

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, 16) and (-2, -10) is 6. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 16
  let x₂ : ℝ := -2
  let y₂ : ℝ := -10
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 6 := by
  sorry

end midpoint_coordinate_sum_l715_71513


namespace sum_of_roots_equation_l715_71500

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a + b = 6) := by
  sorry

end sum_of_roots_equation_l715_71500


namespace ellipse_properties_l715_71557

/-- Properties of an ellipse with equation (x^2 / 25) + (y^2 / 9) = 1 -/
theorem ellipse_properties :
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := 4
  let ellipse := fun (x y : ℝ) ↦ x^2 / 25 + y^2 / 9 = 1
  -- Eccentricity
  (c / a = 0.8) ∧
  -- Foci
  (ellipse (-c) 0 ∧ ellipse c 0) ∧
  -- Vertices
  (ellipse (-a) 0 ∧ ellipse a 0) := by
sorry

end ellipse_properties_l715_71557


namespace sara_golf_balls_l715_71517

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Sara has -/
def sara_dozens : ℕ := 16

/-- The total number of golf balls Sara has -/
def sara_total : ℕ := sara_dozens * dozen

theorem sara_golf_balls : sara_total = 192 := by
  sorry

end sara_golf_balls_l715_71517


namespace correct_price_reduction_equation_l715_71505

/-- Represents the price reduction of a vehicle over two months -/
def price_reduction (initial_price final_price monthly_rate : ℝ) : Prop :=
  initial_price * (1 - monthly_rate)^2 = final_price

/-- Theorem stating the correct equation for the given scenario -/
theorem correct_price_reduction_equation :
  ∃ x : ℝ, price_reduction 23 18.63 x := by
  sorry

end correct_price_reduction_equation_l715_71505


namespace youngsville_population_l715_71540

theorem youngsville_population (P : ℝ) : 
  (P * 1.25 * 0.6 = 513) → P = 684 := by
  sorry

end youngsville_population_l715_71540


namespace plot_perimeter_l715_71504

def rectangular_plot (length width : ℝ) : Prop :=
  length > 0 ∧ width > 0

theorem plot_perimeter (length width : ℝ) :
  rectangular_plot length width →
  length / width = 7 / 5 →
  length * width = 5040 →
  2 * (length + width) = 288 := by
  sorry

end plot_perimeter_l715_71504
