import Mathlib

namespace jelly_beans_initial_amount_l2250_225080

theorem jelly_beans_initial_amount :
  ∀ (initial_amount eaten_amount : ℕ) 
    (num_piles pile_weight : ℕ),
  eaten_amount = 6 →
  num_piles = 3 →
  pile_weight = 10 →
  initial_amount = eaten_amount + num_piles * pile_weight →
  initial_amount = 36 :=
by
  sorry

end jelly_beans_initial_amount_l2250_225080


namespace poll_total_count_l2250_225017

theorem poll_total_count : ∀ (total : ℕ),
  (45 : ℚ) / 100 * total + (8 : ℚ) / 100 * total + (94 : ℕ) = total →
  total = 200 := by
  sorry

end poll_total_count_l2250_225017


namespace chocolate_cost_720_l2250_225042

/-- Calculates the cost of buying a certain number of chocolate candies given the following conditions:
  - A box contains 30 chocolate candies
  - A box costs $10
  - If a customer buys more than 20 boxes, they get a 10% discount
-/
def chocolateCost (numCandies : ℕ) : ℚ :=
  let boxSize := 30
  let boxPrice := 10
  let discountThreshold := 20
  let discountRate := 0.1
  let numBoxes := (numCandies + boxSize - 1) / boxSize  -- Ceiling division
  let totalCost := numBoxes * boxPrice
  if numBoxes > discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

theorem chocolate_cost_720 : chocolateCost 720 = 216 := by
  sorry

end chocolate_cost_720_l2250_225042


namespace expand_product_l2250_225076

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l2250_225076


namespace interval_bound_l2250_225083

-- Define the functions
def f (x : ℝ) := x^4 - 2*x^2
def g (x : ℝ) := 4*x^2 - 8
def h (t x : ℝ) := 4*(t^3 - t)*x - 3*t^4 + 2*t^2

-- State the theorem
theorem interval_bound 
  (t : ℝ) 
  (ht : 0 < |t| ∧ |t| ≤ Real.sqrt 2) 
  (m n : ℝ) 
  (hmn : m ≤ n ∧ Set.Icc m n ⊆ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)) 
  (h_inequality : ∀ x ∈ Set.Icc m n, f x ≥ h t x ∧ h t x ≥ g x) : 
  n - m ≤ Real.sqrt 7 := by
sorry

end interval_bound_l2250_225083


namespace point_on_330_degree_angle_l2250_225026

/-- For any point P (x, y) ≠ (0, 0) on the terminal side of a 330° angle, y/x = -√3/3 -/
theorem point_on_330_degree_angle (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) →  -- Point is not the origin
  (x, y) ∈ {p : ℝ × ℝ | ∃ (r : ℝ), r > 0 ∧ p.1 = r * Real.cos (330 * π / 180) ∧ p.2 = r * Real.sin (330 * π / 180)} →  -- Point is on the terminal side of 330° angle
  y / x = -Real.sqrt 3 / 3 := by
sorry

end point_on_330_degree_angle_l2250_225026


namespace subtraction_equality_l2250_225009

theorem subtraction_equality : 3.65 - 2.27 - 0.48 = 0.90 := by
  sorry

end subtraction_equality_l2250_225009


namespace largest_eight_digit_even_digits_proof_l2250_225012

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_even_digits_proof :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
    n ≤ largest_eight_digit_with_even_digits :=
by sorry

end largest_eight_digit_even_digits_proof_l2250_225012


namespace constant_term_is_135_l2250_225064

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the constant term in the expansion
def constant_term (x : ℝ) : ℝ :=
  binomial 6 2 * 3^2

-- Theorem statement
theorem constant_term_is_135 :
  constant_term = 135 := by sorry

end constant_term_is_135_l2250_225064


namespace consecutive_integers_product_812_sum_57_l2250_225070

theorem consecutive_integers_product_812_sum_57 :
  ∀ x : ℕ, x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_812_sum_57_l2250_225070


namespace sodium_bicarbonate_required_l2250_225048

-- Define the chemical reaction
structure Reaction where
  NaHCO₃ : ℕ
  HCl : ℕ
  NaCl : ℕ
  H₂O : ℕ
  CO₂ : ℕ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.NaHCO₃ = r.HCl ∧ r.NaHCO₃ = r.NaCl ∧ r.NaHCO₃ = r.H₂O ∧ r.NaHCO₃ = r.CO₂

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.HCl = 3 ∧ r.H₂O = 3 ∧ r.CO₂ = 3 ∧ r.NaCl = 3

-- Theorem to prove
theorem sodium_bicarbonate_required (r : Reaction) 
  (h1 : balanced_equation r) (h2 : given_conditions r) : 
  r.NaHCO₃ = 3 := by
  sorry

end sodium_bicarbonate_required_l2250_225048


namespace room_assignment_count_l2250_225074

/-- The number of rooms in the lodge -/
def num_rooms : ℕ := 6

/-- The number of friends checking in -/
def num_friends : ℕ := 7

/-- The maximum number of friends allowed per room -/
def max_per_room : ℕ := 2

/-- The minimum number of unoccupied rooms -/
def min_unoccupied : ℕ := 1

/-- A function that calculates the number of ways to assign friends to rooms -/
def assign_rooms : ℕ := sorry

/-- Theorem stating that the number of ways to assign rooms is 128520 -/
theorem room_assignment_count : assign_rooms = 128520 := by sorry

end room_assignment_count_l2250_225074


namespace train_passing_jogger_time_l2250_225068

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 120)
  (h4 : initial_distance = 180) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 30 := by
sorry

end train_passing_jogger_time_l2250_225068


namespace max_difference_l2250_225089

theorem max_difference (a b : ℝ) : 
  a < 0 → 
  (∀ x, a < x ∧ x < b → (x^2 + 2017*a)*(x + 2016*b) ≥ 0) → 
  b - a ≤ 2017 :=
by sorry

end max_difference_l2250_225089


namespace reading_time_calculation_l2250_225044

theorem reading_time_calculation (pages_per_hour_1 pages_per_hour_2 pages_per_hour_3 : ℕ)
  (total_pages : ℕ) (h1 : pages_per_hour_1 = 21) (h2 : pages_per_hour_2 = 30)
  (h3 : pages_per_hour_3 = 45) (h4 : total_pages = 128) :
  let total_time := (3 * total_pages) / (pages_per_hour_1 + pages_per_hour_2 + pages_per_hour_3)
  total_time = 4 := by
sorry

end reading_time_calculation_l2250_225044


namespace count_switchable_positions_l2250_225022

/-- Represents the number of revolutions a clock hand makes in one hour -/
def revolutions_per_hour (is_minute_hand : Bool) : ℚ :=
  if is_minute_hand then 1 else 1/12

/-- Represents a valid clock position -/
def is_valid_position (hour_pos : ℚ) (minute_pos : ℚ) : Prop :=
  0 ≤ hour_pos ∧ hour_pos < 1 ∧ 0 ≤ minute_pos ∧ minute_pos < 1

/-- Represents a clock position that remains valid when hands are switched -/
def is_switchable_position (t : ℚ) : Prop :=
  is_valid_position (t * revolutions_per_hour false) (t * revolutions_per_hour true) ∧
  is_valid_position (t * revolutions_per_hour true) (t * revolutions_per_hour false)

/-- The main theorem stating the number of switchable positions -/
theorem count_switchable_positions :
  (∃ (S : Finset ℚ), (∀ t ∈ S, is_switchable_position t) ∧ S.card = 143) :=
sorry

end count_switchable_positions_l2250_225022


namespace mary_screws_on_hand_l2250_225037

def screws_needed (sections : ℕ) (screws_per_section : ℕ) : ℕ :=
  sections * screws_per_section

theorem mary_screws_on_hand 
  (sections : ℕ) 
  (screws_per_section : ℕ) 
  (buy_ratio : ℕ) 
  (h1 : sections = 4) 
  (h2 : screws_per_section = 6) 
  (h3 : buy_ratio = 2) :
  ∃ (initial_screws : ℕ), 
    initial_screws + buy_ratio * initial_screws = screws_needed sections screws_per_section ∧ 
    initial_screws = 8 :=
by sorry

end mary_screws_on_hand_l2250_225037


namespace abigail_report_words_l2250_225043

/-- Represents Abigail's report writing scenario -/
structure ReportWriting where
  typing_speed : ℕ  -- words per 30 minutes
  words_written : ℕ
  time_needed : ℕ  -- in minutes

/-- Calculates the total number of words in the report -/
def total_words (r : ReportWriting) : ℕ :=
  r.words_written + r.typing_speed * r.time_needed / 30

/-- Theorem stating that the total words in Abigail's report is 1000 -/
theorem abigail_report_words :
  ∃ (r : ReportWriting), r.typing_speed = 300 ∧ r.words_written = 200 ∧ r.time_needed = 80 ∧ total_words r = 1000 :=
by
  sorry

end abigail_report_words_l2250_225043


namespace fish_in_each_bowl_l2250_225047

theorem fish_in_each_bowl (total_bowls : ℕ) (total_fish : ℕ) (h1 : total_bowls = 261) (h2 : total_fish = 6003) :
  total_fish / total_bowls = 23 := by
  sorry

end fish_in_each_bowl_l2250_225047


namespace unique_right_triangle_l2250_225015

/-- Check if three numbers form a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of line segments -/
def segment_sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]

/-- Theorem: Only one set in segment_sets satisfies the Pythagorean theorem -/
theorem unique_right_triangle : 
  ∃! (a b c : ℕ), (a, b, c) ∈ segment_sets ∧ is_pythagorean_triple a b c :=
by sorry

end unique_right_triangle_l2250_225015


namespace hyperbola_equation_l2250_225062

/-- Given a hyperbola with the specified properties, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 5 / 4
  let c := 5
  (c / a = e) →
  (c^2 = a^2 + b^2) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end hyperbola_equation_l2250_225062


namespace negation_of_exists_square_greater_than_power_of_two_l2250_225040

theorem negation_of_exists_square_greater_than_power_of_two :
  (¬ ∃ (n : ℕ+), n.val ^ 2 > 2 ^ n.val) ↔ ∀ (n : ℕ+), n.val ^ 2 ≤ 2 ^ n.val :=
by sorry

end negation_of_exists_square_greater_than_power_of_two_l2250_225040


namespace bug_returns_probability_l2250_225091

def bug_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | k + 1 => 1/3 * (1 - bug_probability k)

theorem bug_returns_probability :
  bug_probability 12 = 44287 / 177147 :=
by sorry

end bug_returns_probability_l2250_225091


namespace f_difference_l2250_225010

/-- The function f(x) = x^4 + x^2 + 5x^3 -/
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x^3

/-- Theorem: f(5) - f(-5) = 1250 -/
theorem f_difference : f 5 - f (-5) = 1250 := by
  sorry

end f_difference_l2250_225010


namespace exists_N_average_twelve_l2250_225035

theorem exists_N_average_twelve : ∃ N : ℝ, 11 < N ∧ N < 19 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end exists_N_average_twelve_l2250_225035


namespace dice_probability_l2250_225050

/-- The number of sides on each die -/
def sides : ℕ := 15

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The threshold for "low" numbers -/
def low_threshold : ℕ := 10

/-- The number of low outcomes on a single die -/
def low_outcomes : ℕ := low_threshold - 1

/-- The number of high outcomes on a single die -/
def high_outcomes : ℕ := sides - low_outcomes

/-- The probability of rolling a low number on a single die -/
def prob_low : ℚ := low_outcomes / sides

/-- The probability of rolling a high number on a single die -/
def prob_high : ℚ := high_outcomes / sides

/-- The number of ways to choose 3 dice out of 5 -/
def ways_to_choose : ℕ := (num_dice.choose 3)

theorem dice_probability : 
  (ways_to_choose : ℚ) * prob_low^3 * prob_high^2 = 216/625 := by sorry

end dice_probability_l2250_225050


namespace third_and_fourth_terms_equal_21_l2250_225046

def a (n : ℕ) : ℤ := -n^2 + 7*n + 9

theorem third_and_fourth_terms_equal_21 : a 3 = 21 ∧ a 4 = 21 := by
  sorry

end third_and_fourth_terms_equal_21_l2250_225046


namespace vincent_animal_books_l2250_225098

/-- The number of books about animals Vincent bought -/
def num_animal_books : ℕ := sorry

/-- The cost of each book -/
def book_cost : ℕ := 16

/-- The total number of books about outer space and trains -/
def num_other_books : ℕ := 1 + 3

/-- The total amount Vincent spent on books -/
def total_spent : ℕ := 224

theorem vincent_animal_books : 
  num_animal_books = 10 := by sorry

end vincent_animal_books_l2250_225098


namespace root_exists_in_interval_l2250_225053

-- Define the function f(x) = x³ + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end root_exists_in_interval_l2250_225053


namespace intersection_A_complement_B_l2250_225073

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = A ∩ (U \ B) := by sorry

end intersection_A_complement_B_l2250_225073


namespace range_of_m_l2250_225049

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b)
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → a * b = 2 * a + b → a + 2 * b ≥ m^2 - 8 * m) :
  -1 ≤ m ∧ m ≤ 9 := by
  sorry

end range_of_m_l2250_225049


namespace sqrt_inequality_l2250_225077

theorem sqrt_inequality (x : ℝ) :
  (3 - x ≥ 0) → (x + 1 ≥ 0) →
  (Real.sqrt (3 - x) - Real.sqrt (x + 1) > 1/2 ↔ -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8) :=
by sorry

end sqrt_inequality_l2250_225077


namespace sum_of_squares_l2250_225008

theorem sum_of_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eq_a : a + a^2 = 1) (eq_b : b^2 + b^4 = 1) : a^2 + b^2 = 1 := by
  sorry

end sum_of_squares_l2250_225008


namespace tangent_line_at_P_tangent_line_not_at_P_l2250_225051

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the first part
theorem tangent_line_at_P :
  ∃ (l : ℝ → ℝ), (l 1 = -2) ∧ 
  (∀ x : ℝ, l x = -2) ∧
  (∀ x : ℝ, x ≠ 1 → (l x - f x) / (x - 1) ≤ (l 1 - f 1) / (1 - 1)) :=
sorry

-- Theorem for the second part
theorem tangent_line_not_at_P :
  ∃ (l : ℝ → ℝ), (l 1 = -2) ∧ 
  (∀ x : ℝ, 9*x + 4*(l x) - 1 = 0) ∧
  (∃ x₀ : ℝ, x₀ ≠ 1 ∧ 
    (∀ x : ℝ, x ≠ x₀ → (l x - f x) / (x - x₀) ≤ (l x₀ - f x₀) / (x₀ - x₀))) :=
sorry

end tangent_line_at_P_tangent_line_not_at_P_l2250_225051


namespace max_intersection_length_l2250_225020

noncomputable section

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def M : Point := Unit.unit
def N : Point := Unit.unit
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit

-- Define the diameter and its length
def diameter (c : Circle) : ℝ := 2

-- Define the property that MN is a diameter
def is_diameter (c : Circle) (m n : Point) : Prop := True

-- Define A as the midpoint of the semicircular arc
def is_midpoint_arc (c : Circle) (m n a : Point) : Prop := True

-- Define the length of MB
def length_MB : ℝ := 4/7

-- Define C as a point on the other semicircular arc
def on_other_arc (c : Circle) (m n c : Point) : Prop := True

-- Define the intersections of MN with AC and BC
def intersection_AC_MN (c : Circle) (m n a c : Point) : Point := Unit.unit
def intersection_BC_MN (c : Circle) (m n b c : Point) : Point := Unit.unit

-- Define the length of the line segment formed by the intersections
def length_intersections (p q : Point) : ℝ := 0

-- Theorem statement
theorem max_intersection_length (c : Circle) :
  is_diameter c M N →
  is_midpoint_arc c M N A →
  length_MB = 4/7 →
  on_other_arc c M N C →
  ∃ (d : ℝ), d = 10 - 7 * Real.sqrt 3 ∧
    ∀ (V W : Point),
      V = intersection_AC_MN c M N A C →
      W = intersection_BC_MN c M N B C →
      length_intersections V W ≤ d :=
sorry

end

end max_intersection_length_l2250_225020


namespace train_passengers_l2250_225095

theorem train_passengers (P : ℕ) : 
  (((P - P / 3 + 280) / 2 + 12) = 248) → P = 288 := by
  sorry

end train_passengers_l2250_225095


namespace product_equality_l2250_225090

theorem product_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a * b + b * c + a * c) * ((a * b)⁻¹ + (b * c)⁻¹ + (a * c)⁻¹) = 
  (a * b + b * c + a * c)^2 / (a * b * c) := by
  sorry

end product_equality_l2250_225090


namespace no_right_angle_in_sequence_l2250_225056

/-- Represents a triangle with three angles -/
structure Triangle where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { angleA := t.angleA, angleB := t.angleB, angleC := t.angleC }

/-- The original triangle ABC -/
def originalTriangle : Triangle :=
  { angleA := 59, angleB := 61, angleC := 60 }

/-- Generates the nth triangle in the sequence -/
def nthTriangle (n : ℕ) : Triangle :=
  match n with
  | 0 => originalTriangle
  | n+1 => nextTriangle (nthTriangle n)

theorem no_right_angle_in_sequence :
  ∀ n : ℕ, (nthTriangle n).angleA ≠ 90 ∧ (nthTriangle n).angleB ≠ 90 ∧ (nthTriangle n).angleC ≠ 90 :=
sorry

end no_right_angle_in_sequence_l2250_225056


namespace parabola_max_value_l2250_225019

/-- A parabola that opens downward and has its vertex at (2, -3) has a maximum value of -3 -/
theorem parabola_max_value (a b c : ℝ) (h_downward : a < 0) 
  (h_vertex : ∀ x, a * x^2 + b * x + c ≤ a * 2^2 + b * 2 + c) 
  (h_vertex_y : a * 2^2 + b * 2 + c = -3) : 
  ∀ x, a * x^2 + b * x + c ≤ -3 :=
sorry

end parabola_max_value_l2250_225019


namespace cyclist_average_speed_l2250_225067

/-- Calculates the average speed of a cyclist's trip given two segments with different speeds -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 11) (h3 : v1 = 11) (h4 : v2 = 8) :
  (d1 + d2) / ((d1 / v1) + (d2 / v2)) = 1664 / 185 := by
  sorry

end cyclist_average_speed_l2250_225067


namespace divisibility_theorem_l2250_225032

theorem divisibility_theorem (a b c d u : ℤ) 
  (h1 : u ∣ (a * c)) 
  (h2 : u ∣ (b * c + a * d)) 
  (h3 : u ∣ (b * d)) : 
  (u ∣ (b * c)) ∧ (u ∣ (a * d)) := by
  sorry

end divisibility_theorem_l2250_225032


namespace f_f_has_four_roots_l2250_225001

def f (x : ℝ) := x^2 - 3*x + 2

theorem f_f_has_four_roots :
  ∃! (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, f (f x) = 0) ∧ (∀ y, f (f y) = 0 → y ∈ s) :=
sorry

end f_f_has_four_roots_l2250_225001


namespace two_digit_number_divisible_by_55_l2250_225052

theorem two_digit_number_divisible_by_55 (a b : ℕ) : 
  (a ≥ 1 ∧ a ≤ 9) →  -- 'a' is a single digit (tens place)
  (b ≥ 0 ∧ b ≤ 9) →  -- 'b' is a single digit (units place)
  (10 * a + b) % 55 = 0 →  -- number is divisible by 55
  (∀ (x y : ℕ), (x ≥ 1 ∧ x ≤ 9) → (y ≥ 0 ∧ y ≤ 9) → (10 * x + y) % 55 = 0 → x * y ≤ 15) →  -- greatest possible value of b × a is 15
  10 * a + b = 65 :=
by sorry

end two_digit_number_divisible_by_55_l2250_225052


namespace logarithm_sum_simplification_l2250_225086

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + Real.log 4 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + Real.log 4 / Real.log 8 + 1) +
  1 / (1 + (Real.log 5 / Real.log 15 + Real.log 3 / Real.log 15)) = 3/2 :=
by sorry

end logarithm_sum_simplification_l2250_225086


namespace sum_odd_integers_l2250_225082

theorem sum_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : n^2 = 4900 := by
  sorry

end sum_odd_integers_l2250_225082


namespace units_digit_of_7_power_2006_l2250_225096

theorem units_digit_of_7_power_2006 : ∃ n : ℕ, 7^2006 ≡ 9 [ZMOD 10] := by sorry

end units_digit_of_7_power_2006_l2250_225096


namespace expression_equals_four_l2250_225094

theorem expression_equals_four :
  (8 : ℝ) ^ (1/3) + (1/3)⁻¹ - 2 * Real.cos (30 * π / 180) + |1 - Real.sqrt 3| = 4 := by
  sorry

end expression_equals_four_l2250_225094


namespace min_max_abs_quadratic_minus_linear_l2250_225029

theorem min_max_abs_quadratic_minus_linear (y : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → |x^2 - x*y| ≤ 0 ∧
  (∀ (y : ℝ), ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - x*y| ≥ 0) :=
sorry

end min_max_abs_quadratic_minus_linear_l2250_225029


namespace cost_of_candies_l2250_225054

def candies_per_box : ℕ := 30
def cost_per_box : ℕ := 8
def total_candies : ℕ := 450

theorem cost_of_candies :
  (total_candies / candies_per_box) * cost_per_box = 120 := by
sorry

end cost_of_candies_l2250_225054


namespace floor_sqrt_24_squared_l2250_225084

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end floor_sqrt_24_squared_l2250_225084


namespace negation_of_forall_geq_zero_is_exists_lt_zero_l2250_225036

theorem negation_of_forall_geq_zero_is_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 + x ≥ 0) ↔ (∃ x : ℝ, x^2 + x < 0) := by sorry

end negation_of_forall_geq_zero_is_exists_lt_zero_l2250_225036


namespace sum_c_plus_d_l2250_225078

theorem sum_c_plus_d (a b c d : ℝ) 
  (h1 : a + b = 5)
  (h2 : b + c = 6)
  (h3 : a + d = 2) :
  c + d = 3 := by
  sorry

end sum_c_plus_d_l2250_225078


namespace oil_quantity_function_correct_l2250_225055

/-- Represents the remaining oil quantity in a tank as a function of time -/
def Q (t : ℝ) : ℝ := 20 - 0.2 * t

/-- The initial oil quantity in the tank -/
def initial_quantity : ℝ := 20

/-- The rate at which oil flows out of the tank (in liters per minute) -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct :
  ∀ t : ℝ, t ≥ 0 →
  Q t = initial_quantity - flow_rate * t ∧
  Q t ≥ 0 ∧
  (Q t = 0 → t = initial_quantity / flow_rate) :=
sorry

end oil_quantity_function_correct_l2250_225055


namespace materik_position_l2250_225011

def Alphabet : Finset Char := {'A', 'E', 'I', 'K', 'M', 'R', 'T'}

def Word := List Char

def isValidWord (w : Word) : Prop :=
  w.length = 7 ∧ w.toFinset = Alphabet

def alphabeticalOrder (order : List Char) : Prop :=
  order.length = 7 ∧ order.toFinset = Alphabet

def wordPosition (w : Word) (order : List Char) : ℕ :=
  sorry

theorem materik_position 
  (order : List Char) 
  (h_order : alphabeticalOrder order) 
  (h_metrika : wordPosition ['M', 'E', 'T', 'R', 'I', 'K', 'A'] order = 3634) :
  wordPosition ['M', 'A', 'T', 'E', 'R', 'I', 'K'] order = 3745 :=
sorry

end materik_position_l2250_225011


namespace average_speed_round_trip_l2250_225093

/-- Given uphill speed V₁ and downhill speed V₂, 
    the average speed for a round trip is (2 * V₁ * V₂) / (V₁ + V₂) -/
theorem average_speed_round_trip (V₁ V₂ : ℝ) (h₁ : V₁ > 0) (h₂ : V₂ > 0) :
  let s : ℝ := 1  -- Assume unit distance for simplicity
  let t_up : ℝ := s / V₁
  let t_down : ℝ := s / V₂
  let total_distance : ℝ := 2 * s
  let total_time : ℝ := t_up + t_down
  total_distance / total_time = (2 * V₁ * V₂) / (V₁ + V₂) :=
by sorry

end average_speed_round_trip_l2250_225093


namespace hat_problem_inconsistent_l2250_225028

/-- Represents the number of hats of each color --/
structure HatCounts where
  blue : ℕ
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Checks if the given hat counts satisfy the problem conditions --/
def satisfies_conditions (counts : HatCounts) : Prop :=
  counts.blue + counts.green + counts.red + counts.yellow = 150 ∧
  counts.blue = 2 * counts.green ∧
  8 * counts.blue + 10 * counts.green + 12 * counts.red + 15 * counts.yellow = 1280

/-- Theorem stating the inconsistency in the problem --/
theorem hat_problem_inconsistent : 
  ∀ (counts : HatCounts), satisfies_conditions counts → counts.red = 0 ∧ counts.yellow = 0 := by
  sorry

#check hat_problem_inconsistent

end hat_problem_inconsistent_l2250_225028


namespace paco_ate_five_sweet_cookies_l2250_225018

/-- Represents the number of cookies Paco had and ate -/
structure CookieCount where
  initial_sweet : Nat
  initial_salty : Nat
  eaten_salty : Nat
  sweet_salty_difference : Nat

/-- Calculates the number of sweet cookies Paco ate -/
def sweet_cookies_eaten (c : CookieCount) : Nat :=
  c.eaten_salty + c.sweet_salty_difference

/-- Theorem: Paco ate 5 sweet cookies -/
theorem paco_ate_five_sweet_cookies (c : CookieCount)
  (h1 : c.initial_sweet = 37)
  (h2 : c.initial_salty = 11)
  (h3 : c.eaten_salty = 2)
  (h4 : c.sweet_salty_difference = 3) :
  sweet_cookies_eaten c = 5 := by
  sorry

end paco_ate_five_sweet_cookies_l2250_225018


namespace democrat_count_l2250_225061

theorem democrat_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 870 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 145 := by
  sorry

end democrat_count_l2250_225061


namespace class_average_problem_l2250_225034

theorem class_average_problem (n₁ n₂ : ℕ) (avg₂ avg_combined : ℚ) :
  n₁ = 30 →
  n₂ = 50 →
  avg₂ = 60 →
  avg_combined = 52.5 →
  ∃ avg₁ : ℚ, avg₁ = 40 ∧ (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = avg_combined :=
by sorry

end class_average_problem_l2250_225034


namespace expression_evaluation_l2250_225069

theorem expression_evaluation : 
  (Real.sqrt 3 - 4 * Real.sin (20 * π / 180) + 8 * (Real.sin (20 * π / 180))^3) / 
  (2 * Real.sin (20 * π / 180) * Real.sin (480 * π / 180)) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end expression_evaluation_l2250_225069


namespace divisibility_implies_inequality_l2250_225057

theorem divisibility_implies_inequality (k m : ℕ+) (h1 : k > m) 
  (h2 : (k^3 - m^3) ∣ (k * m * (k^2 - m^2))) : 
  (k - m)^3 > 3 * k * m := by
  sorry

end divisibility_implies_inequality_l2250_225057


namespace simplify_nested_expression_l2250_225006

theorem simplify_nested_expression (x : ℝ) : 1 - (1 - (1 + (1 - (1 + (1 - x))))) = 3 - x := by
  sorry

end simplify_nested_expression_l2250_225006


namespace vector_problem_l2250_225024

/-- Given two vectors a and b in ℝ², proves that if a is collinear with b and their dot product is -10, then b is equal to (-4, 2) -/
theorem vector_problem (a b : ℝ × ℝ) : 
  a = (2, -1) → 
  (∃ k : ℝ, b = k • a) → 
  a.1 * b.1 + a.2 * b.2 = -10 → 
  b = (-4, 2) := by
sorry

end vector_problem_l2250_225024


namespace product_of_fractions_equals_64_l2250_225004

theorem product_of_fractions_equals_64 :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end product_of_fractions_equals_64_l2250_225004


namespace expand_and_simplify_l2250_225097

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 4) * (8 / x - 15 * x^3 + 6 * x) = 6 / x - 45 / 4 * x^3 + 9 / 2 * x := by
  sorry

end expand_and_simplify_l2250_225097


namespace rectangle_circle_tangent_l2250_225059

/-- Given a circle with radius 6 cm tangent to two shorter sides and one longer side of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the shorter side of the rectangle is 12 cm. -/
theorem rectangle_circle_tangent (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ) :
  circle_radius = 6 →
  rectangle_area = 3 * circle_area →
  circle_area = Real.pi * circle_radius^2 →
  (12 : ℝ) = 2 * circle_radius :=
by sorry

end rectangle_circle_tangent_l2250_225059


namespace expression_factorization_l2250_225013

theorem expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by sorry

end expression_factorization_l2250_225013


namespace units_digit_of_7_cubed_l2250_225002

theorem units_digit_of_7_cubed (n : ℕ) : n = 7^3 → n % 10 = 3 := by
  sorry

end units_digit_of_7_cubed_l2250_225002


namespace complement_of_A_in_U_l2250_225071

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 5} := by sorry

end complement_of_A_in_U_l2250_225071


namespace solve_for_y_l2250_225045

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end solve_for_y_l2250_225045


namespace liam_needed_one_more_correct_answer_l2250_225031

/-- Represents the number of questions Liam answered correctly in each category -/
structure CorrectAnswers where
  programming : ℕ
  dataStructures : ℕ
  algorithms : ℕ

/-- Calculates the total number of correct answers -/
def totalCorrect (answers : CorrectAnswers) : ℕ :=
  answers.programming + answers.dataStructures + answers.algorithms

/-- Represents the examination structure and Liam's performance -/
structure Examination where
  totalQuestions : ℕ
  programmingQuestions : ℕ
  dataStructuresQuestions : ℕ
  algorithmsQuestions : ℕ
  passingPercentage : ℚ
  correctAnswers : CorrectAnswers

/-- Theorem stating that Liam needed 1 more correct answer to pass -/
theorem liam_needed_one_more_correct_answer (exam : Examination)
  (h1 : exam.totalQuestions = 50)
  (h2 : exam.programmingQuestions = 15)
  (h3 : exam.dataStructuresQuestions = 20)
  (h4 : exam.algorithmsQuestions = 15)
  (h5 : exam.passingPercentage = 65 / 100)
  (h6 : exam.correctAnswers.programming = 12)
  (h7 : exam.correctAnswers.dataStructures = 10)
  (h8 : exam.correctAnswers.algorithms = 10) :
  ⌈exam.totalQuestions * exam.passingPercentage⌉ - totalCorrect exam.correctAnswers = 1 := by
  sorry


end liam_needed_one_more_correct_answer_l2250_225031


namespace dorchester_puppies_washed_l2250_225087

/-- Calculates the number of puppies washed given the total earnings, base pay, and rate per puppy -/
def puppies_washed (total_earnings base_pay rate_per_puppy : ℚ) : ℚ :=
  (total_earnings - base_pay) / rate_per_puppy

/-- Proves that Dorchester washed 16 puppies on Wednesday -/
theorem dorchester_puppies_washed :
  puppies_washed 76 40 (9/4) = 16 := by
  sorry

end dorchester_puppies_washed_l2250_225087


namespace fraction_zero_implies_x_values_l2250_225060

theorem fraction_zero_implies_x_values (x : ℝ) : 
  (x ^ 2 - 4) / x = 0 → x = 2 ∨ x = -2 :=
by sorry

end fraction_zero_implies_x_values_l2250_225060


namespace lemonade_glasses_count_l2250_225003

/-- The number of glasses of lemonade that can be served from one pitcher -/
def glasses_per_pitcher : ℕ := 5

/-- The number of pitchers of lemonade prepared -/
def number_of_pitchers : ℕ := 6

/-- The total number of glasses of lemonade that can be served -/
def total_glasses : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_glasses_count : total_glasses = 30 := by
  sorry

end lemonade_glasses_count_l2250_225003


namespace five_people_seven_chairs_l2250_225063

/-- The number of ways to arrange n people in k chairs, where the first person
    cannot sit in m specific chairs. -/
def seating_arrangements (n k m : ℕ) : ℕ :=
  (k - m) * (k - 1).factorial / (k - n).factorial

/-- The problem statement -/
theorem five_people_seven_chairs : seating_arrangements 5 7 2 = 1800 := by
  sorry

end five_people_seven_chairs_l2250_225063


namespace intersection_of_A_and_B_l2250_225075

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 + 5}
def B : Set (ℝ × ℝ) := {p | p.2 = 1 - 2 * p.1}

theorem intersection_of_A_and_B : A ∩ B = {(-1, 3)} := by
  sorry

end intersection_of_A_and_B_l2250_225075


namespace amc_12_score_problem_l2250_225085

theorem amc_12_score_problem (total_problems : Nat) (attempted_problems : Nat) 
  (correct_points : Nat) (incorrect_points : Nat) (unanswered_points : Nat) 
  (unanswered_count : Nat) (min_score : Nat) :
  total_problems = 30 →
  attempted_problems = 26 →
  correct_points = 7 →
  incorrect_points = 0 →
  unanswered_points = 1 →
  unanswered_count = 4 →
  min_score = 150 →
  ∃ (correct_count : Nat), 
    correct_count * correct_points + 
    (attempted_problems - correct_count) * incorrect_points + 
    unanswered_count * unanswered_points ≥ min_score ∧
    correct_count = 21 ∧
    ∀ (x : Nat), x < 21 → 
      x * correct_points + 
      (attempted_problems - x) * incorrect_points + 
      unanswered_count * unanswered_points < min_score :=
by sorry

end amc_12_score_problem_l2250_225085


namespace M_on_y_axis_coordinates_l2250_225007

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point on the y-axis -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- The point M with coordinates (m+1, m+3) -/
def M (m : ℝ) : Point :=
  { x := m + 1
    y := m + 3 }

/-- Theorem: If M(m+1, m+3) is on the y-axis, then its coordinates are (0, 2) -/
theorem M_on_y_axis_coordinates :
  ∀ m : ℝ, on_y_axis (M m) → M m = { x := 0, y := 2 } := by
  sorry

end M_on_y_axis_coordinates_l2250_225007


namespace exponent_multiplication_specific_exponent_multiplication_l2250_225027

theorem exponent_multiplication (a b c : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) :=
by sorry

theorem specific_exponent_multiplication : (10 : ℝ) ^ 10000 * (10 : ℝ) ^ 8000 = (10 : ℝ) ^ 18000 :=
by sorry

end exponent_multiplication_specific_exponent_multiplication_l2250_225027


namespace fraction_problem_l2250_225099

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  (0.4 * N = 180) → 
  (f * (1/3) * (2/5) * N = 15) → 
  f = 1/4 := by
sorry

end fraction_problem_l2250_225099


namespace calculate_second_discount_l2250_225041

/-- Given an article with a list price and two successive discounts, 
    calculate the second discount percentage. -/
theorem calculate_second_discount 
  (list_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (h1 : list_price = 70) 
  (h2 : first_discount = 10) 
  (h3 : final_price = 61.74) : 
  ∃ (second_discount : ℝ), 
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧ 
    second_discount = 2 := by
  sorry

end calculate_second_discount_l2250_225041


namespace tangent_line_to_circle_l2250_225025

/-- The equation of a line passing through (1,1) and tangent to the circle x^2 - 2x + y^2 = 0 is y = 1 -/
theorem tangent_line_to_circle (x y : ℝ) : 
  (∃ k : ℝ, y - 1 = k * (x - 1)) ∧ 
  (x^2 - 2*x + y^2 = 0 → (x - 1)^2 + (y - 0)^2 = 1) →
  y = 1 :=
sorry

end tangent_line_to_circle_l2250_225025


namespace hidden_lattice_points_l2250_225081

theorem hidden_lattice_points (n : ℕ+) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n ∧ j < n → Nat.gcd (Int.toNat (a + i)) (Int.toNat (b + j)) > 1 := by
  sorry

end hidden_lattice_points_l2250_225081


namespace complex_product_polar_form_l2250_225072

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the problem statement
theorem complex_product_polar_form :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  (4 * cis (25 * π / 180)) * (-3 * cis (48 * π / 180)) = r * cis θ ∧
  r = 12 ∧ θ = 253 * π / 180 :=
by sorry

end complex_product_polar_form_l2250_225072


namespace square_root_of_nine_three_is_square_root_of_nine_l2250_225023

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 → x = 3 ∨ x = -3 := by
  sorry

theorem three_is_square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by
  sorry

end square_root_of_nine_three_is_square_root_of_nine_l2250_225023


namespace expression_value_l2250_225000

theorem expression_value : 
  let x : ℝ := 26
  let y : ℝ := 3 * x / 2
  let z : ℝ := 11
  (x - (y - z)) - ((x - y) - z) = 22 := by
sorry

end expression_value_l2250_225000


namespace emilys_weight_l2250_225030

/-- Given Heather's weight and the difference between Heather and Emily's weights,
    prove that Emily's weight is 9 pounds. -/
theorem emilys_weight (heathers_weight : ℕ) (weight_difference : ℕ)
  (hw : heathers_weight = 87)
  (diff : weight_difference = 78)
  : heathers_weight - weight_difference = 9 := by
  sorry

#check emilys_weight

end emilys_weight_l2250_225030


namespace proportion_solution_l2250_225092

theorem proportion_solution (x : ℝ) : (0.75 / x = 10 / 8) → x = 0.6 := by
  sorry

end proportion_solution_l2250_225092


namespace doughnut_sharing_l2250_225079

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens Samuel bought -/
def samuel_dozens : ℕ := 2

/-- The number of dozens Cathy bought -/
def cathy_dozens : ℕ := 3

/-- The number of doughnuts each person received -/
def doughnuts_per_person : ℕ := 6

/-- The number of people who bought the doughnuts (Samuel and Cathy) -/
def buyers : ℕ := 2

theorem doughnut_sharing :
  let total_doughnuts := samuel_dozens * dozen + cathy_dozens * dozen
  let total_people := total_doughnuts / doughnuts_per_person
  total_people - buyers = 8 := by
  sorry

end doughnut_sharing_l2250_225079


namespace division_of_fractions_l2250_225058

theorem division_of_fractions : (3 : ℚ) / (6 / 11) = 11 / 2 := by
  sorry

end division_of_fractions_l2250_225058


namespace sum_10_to_16_l2250_225065

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q
  sum_2_4 : a 2 + a 4 = 32
  sum_6_8 : a 6 + a 8 = 16

/-- The sum of the 10th, 12th, 14th, and 16th terms equals 12 -/
theorem sum_10_to_16 (seq : GeometricSequence) :
  seq.a 10 + seq.a 12 + seq.a 14 + seq.a 16 = 12 := by
  sorry

end sum_10_to_16_l2250_225065


namespace overlap_area_theorem_l2250_225016

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side : ℝ)
  (rotation : ℝ)

/-- Calculates the area of the polygon formed by overlapping rotated squares -/
def overlappingArea (sheets : List Sheet) : ℝ :=
  sorry

theorem overlap_area_theorem : 
  let sheets : List Sheet := [
    { side := 8, rotation := 0 },
    { side := 8, rotation := 15 },
    { side := 8, rotation := 45 },
    { side := 8, rotation := 75 }
  ]
  overlappingArea sheets = 512 := by sorry

end overlap_area_theorem_l2250_225016


namespace solution_of_system_l2250_225039

/-- Given a system of equations, prove that the solutions are (2, 1) and (2/5, -1/5) -/
theorem solution_of_system :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 2 ∧ y₁ = 1) ∧
    (x₂ = 2/5 ∧ y₂ = -1/5) ∧
    (∀ x y : ℝ,
      (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧
       5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end solution_of_system_l2250_225039


namespace geometric_progression_fifth_term_l2250_225038

theorem geometric_progression_fifth_term 
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4 : ℝ))
  (h₂ : a₂ = 2^(1/5 : ℝ))
  (h₃ : a₃ = 2^(1/6 : ℝ))
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₅ : ℝ, a₅ = 2^(11/60 : ℝ) ∧ 
    ∃ a₄ : ℝ, a₄ = a₃ * (a₂ / a₁) ∧ a₅ = a₄ * (a₂ / a₁) :=
by sorry

end geometric_progression_fifth_term_l2250_225038


namespace negation_of_p_l2250_225005

def p : Prop := ∀ x : ℝ, Real.sqrt (2 - x) < 0

theorem negation_of_p : ¬p ↔ ∃ x₀ : ℝ, Real.sqrt (2 - x₀) ≥ 0 := by
  sorry

end negation_of_p_l2250_225005


namespace symmetry_x_axis_coordinates_l2250_225014

/-- Two points are symmetric with respect to the x-axis if they have the same x-coordinate
    and opposite y-coordinates -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given that P(-2, 3) is symmetric to Q(a, b) with respect to the x-axis,
    prove that a = -2 and b = -3 -/
theorem symmetry_x_axis_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let Q : ℝ × ℝ := (a, b)
  symmetric_x_axis P Q → a = -2 ∧ b = -3 := by
  sorry


end symmetry_x_axis_coordinates_l2250_225014


namespace second_derivative_value_l2250_225088

def f (q : ℝ) : ℝ := 3 * q - 3

theorem second_derivative_value (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end second_derivative_value_l2250_225088


namespace parallel_line_through_point_l2250_225033

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- The main theorem
theorem parallel_line_through_point :
  let A : Point2D := ⟨-1, 0⟩
  let l1 : Line2D := ⟨2, -1, 1⟩
  let l2 : Line2D := ⟨2, -1, 2⟩
  pointOnLine A l2 ∧ linesParallel l1 l2 :=
by sorry

end parallel_line_through_point_l2250_225033


namespace train_length_calculation_l2250_225021

theorem train_length_calculation (platform_crossing_time platform_length signal_crossing_time : ℝ) 
  (h1 : platform_crossing_time = 27)
  (h2 : platform_length = 150.00000000000006)
  (h3 : signal_crossing_time = 18) :
  ∃ train_length : ℝ, train_length = 300.0000000000001 ∧
    platform_crossing_time * (train_length / signal_crossing_time) = train_length + platform_length :=
by
  sorry

end train_length_calculation_l2250_225021


namespace gcd_987654_876543_l2250_225066

theorem gcd_987654_876543 : Nat.gcd 987654 876543 = 3 := by
  sorry

end gcd_987654_876543_l2250_225066
