import Mathlib

namespace NUMINAMATH_GPT_probability_of_different_suits_l1167_116784

-- Let’s define the parameters of the problem
def total_cards : ℕ := 104
def first_card_remaining : ℕ := 103
def same_suit_cards : ℕ := 26
def different_suit_cards : ℕ := first_card_remaining - same_suit_cards

-- The probability that the two cards drawn are of different suits
def probability_different_suits : ℚ := different_suit_cards / first_card_remaining

-- The main statement to prove
theorem probability_of_different_suits :
  probability_different_suits = 78 / 103 :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_probability_of_different_suits_l1167_116784


namespace NUMINAMATH_GPT_solve_equation_l1167_116705

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 2) = (x - 2) → (x = 2 ∨ x = 1 / 3) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1167_116705


namespace NUMINAMATH_GPT_number_of_packs_of_cake_l1167_116795

-- Define the total number of packs of groceries
def total_packs : ℕ := 14

-- Define the number of packs of cookies
def packs_of_cookies : ℕ := 2

-- Define the number of packs of cake as total packs minus packs of cookies
def packs_of_cake : ℕ := total_packs - packs_of_cookies

theorem number_of_packs_of_cake :
  packs_of_cake = 12 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_packs_of_cake_l1167_116795


namespace NUMINAMATH_GPT_incorrect_statements_are_1_2_4_l1167_116773

theorem incorrect_statements_are_1_2_4:
    let statements := ["Inductive reasoning and analogical reasoning both involve reasoning from specific to general.",
                       "When making an analogy, it is more appropriate to use triangles in a plane and parallelepipeds in space as the objects of analogy.",
                       "'All multiples of 9 are multiples of 3, if a number m is a multiple of 9, then m must be a multiple of 3' is an example of syllogistic reasoning.",
                       "In deductive reasoning, as long as it follows the form of deductive reasoning, the conclusion is always correct."]
    let incorrect_statements := {1, 2, 4}
    incorrect_statements = {i | i ∈ [1, 2, 3, 4] ∧
                             ((i = 1 → ¬(∃ s, s ∈ statements ∧ s = statements[0])) ∧ 
                              (i = 2 → ¬(∃ s, s ∈ statements ∧ s = statements[1])) ∧ 
                              (i = 3 → ∃ s, s ∈ statements ∧ s = statements[2]) ∧ 
                              (i = 4 → ¬(∃ s, s ∈ statements ∧ s = statements[3])))} :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statements_are_1_2_4_l1167_116773


namespace NUMINAMATH_GPT_school_choir_robe_cost_l1167_116770

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end NUMINAMATH_GPT_school_choir_robe_cost_l1167_116770


namespace NUMINAMATH_GPT_Lisa_total_spoons_l1167_116744

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end NUMINAMATH_GPT_Lisa_total_spoons_l1167_116744


namespace NUMINAMATH_GPT_blue_balls_prob_l1167_116762

def prob_same_color (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

theorem blue_balls_prob {n : ℕ} (h : prob_same_color n = 1 / 2) : n = 1 ∨ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_blue_balls_prob_l1167_116762


namespace NUMINAMATH_GPT_speed_second_half_l1167_116703

theorem speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) :
    total_time = 12 → first_half_speed = 35 → total_distance = 560 → 
    (280 / (12 - (280 / 35)) = 70) :=
by
  intros ht hf hd
  sorry

end NUMINAMATH_GPT_speed_second_half_l1167_116703


namespace NUMINAMATH_GPT_mans_rate_in_still_water_l1167_116788

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h1 : V_m + V_s = 20)
  (h2 : V_m - V_s = 4) :
  V_m = 12 :=
by
  sorry

end NUMINAMATH_GPT_mans_rate_in_still_water_l1167_116788


namespace NUMINAMATH_GPT_find_m_l1167_116783

theorem find_m (x y m : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : 3 * x - 4 * (m - 1) * y + 30 = 0) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1167_116783


namespace NUMINAMATH_GPT_number_of_lemons_l1167_116775

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_lemons_l1167_116775


namespace NUMINAMATH_GPT_two_digit_number_digits_34_l1167_116772

theorem two_digit_number_digits_34 :
  let x := (34 / 99.0)
  ∃ n : ℕ, n = 34 ∧ (48 * x - 48 * 0.34 = 0.2) := 
by
  let x := (34.0 / 99.0)
  use 34
  sorry

end NUMINAMATH_GPT_two_digit_number_digits_34_l1167_116772


namespace NUMINAMATH_GPT_Chris_age_l1167_116745

theorem Chris_age 
  (a b c : ℝ)
  (h1 : a + b + c = 36)
  (h2 : c - 5 = a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 15.5454545454545 :=
by
  sorry

end NUMINAMATH_GPT_Chris_age_l1167_116745


namespace NUMINAMATH_GPT_range_of_m_l1167_116726

theorem range_of_m (m : ℝ) :
  let p := (2 < m ∧ m < 4)
  let q := (m > 1 ∧ 4 - 4 * m < 0)
  (¬ (p ∧ q) ∧ (p ∨ q)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 4) :=
by intros p q h
   let p := 2 < m ∧ m < 4
   let q := m > 1 ∧ 4 - 4 * m < 0
   sorry

end NUMINAMATH_GPT_range_of_m_l1167_116726


namespace NUMINAMATH_GPT_math_problem_l1167_116752

theorem math_problem (x : ℤ) (h : x = 9) :
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1167_116752


namespace NUMINAMATH_GPT_proof_equivalent_l1167_116760

variables {α : Type*} [Field α]

theorem proof_equivalent (a b c d e f : α)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 :=
by sorry

end NUMINAMATH_GPT_proof_equivalent_l1167_116760


namespace NUMINAMATH_GPT_shirts_made_today_l1167_116723

def shirts_per_minute : ℕ := 8
def working_minutes : ℕ := 2

theorem shirts_made_today (h1 : shirts_per_minute = 8) (h2 : working_minutes = 2) : shirts_per_minute * working_minutes = 16 := by
  sorry

end NUMINAMATH_GPT_shirts_made_today_l1167_116723


namespace NUMINAMATH_GPT_least_possible_b_l1167_116757

theorem least_possible_b (a b : ℕ) (h1 : a + b = 120) (h2 : (Prime a ∨ ∃ p : ℕ, Prime p ∧ a = 2 * p)) (h3 : Prime b) (h4 : a > b) : b = 7 :=
sorry

end NUMINAMATH_GPT_least_possible_b_l1167_116757


namespace NUMINAMATH_GPT_eq_of_fraction_eq_l1167_116715

variable {R : Type*} [Field R]

theorem eq_of_fraction_eq (a b : R) (h : (1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b))) : a = b :=
sorry

end NUMINAMATH_GPT_eq_of_fraction_eq_l1167_116715


namespace NUMINAMATH_GPT_coupons_per_coloring_book_l1167_116718

theorem coupons_per_coloring_book 
  (initial_books : ℝ) (books_sold : ℝ) (coupons_used : ℝ)
  (h1 : initial_books = 40) (h2 : books_sold = 20) (h3 : coupons_used = 80) : 
  (coupons_used / (initial_books - books_sold) = 4) :=
by 
  simp [*, sub_eq_add_neg]
  sorry

end NUMINAMATH_GPT_coupons_per_coloring_book_l1167_116718


namespace NUMINAMATH_GPT_radius_of_circle_eqn_zero_l1167_116763

def circle_eqn (x y : ℝ) := x^2 + 8*x + y^2 - 4*y + 20 = 0

theorem radius_of_circle_eqn_zero :
  ∀ x y : ℝ, circle_eqn x y → ∃ r : ℝ, r = 0 :=
by
  intros x y h
  -- Sorry to skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_radius_of_circle_eqn_zero_l1167_116763


namespace NUMINAMATH_GPT_find_m_l1167_116761

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end NUMINAMATH_GPT_find_m_l1167_116761


namespace NUMINAMATH_GPT_garden_area_remaining_l1167_116756

variable (d : ℕ) (w : ℕ) (t : ℕ)

theorem garden_area_remaining (r : Real) (A_circle : Real) 
                              (A_path : Real) (A_remaining : Real) :
  r = 10 →
  A_circle = 100 * Real.pi →
  A_path = 66.66 * Real.pi - 50 * Real.sqrt 3 →
  A_remaining = 33.34 * Real.pi + 50 * Real.sqrt 3 :=
by
  -- Given the radius of the garden
  let r := (d : Real) / 2
  -- Calculate the total area of the garden
  let A_circle := Real.pi * r^2
  -- Area covered by the path computed using circular segments
  let A_path := 66.66 * Real.pi - 50 * Real.sqrt 3
  -- Remaining garden area
  let A_remaining := A_circle - A_path
  -- Statement to prove correct
  sorry 

end NUMINAMATH_GPT_garden_area_remaining_l1167_116756


namespace NUMINAMATH_GPT_problem_area_triangle_PNT_l1167_116753

noncomputable def area_triangle_PNT (PQ QR x : ℝ) : ℝ :=
  let PS := Real.sqrt (PQ^2 + QR^2)
  let PN := PS / 2
  let area := (PN * Real.sqrt (61 - x^2)) / 4
  area

theorem problem_area_triangle_PNT :
  ∀ (PQ QR : ℝ) (x : ℝ), PQ = 10 → QR = 12 → 0 ≤ x ∧ x ≤ 10 → area_triangle_PNT PQ QR x = 
  (Real.sqrt (244) * Real.sqrt (61 - x^2)) / 4 :=
by
  intros PQ QR x hPQ hQR hx
  sorry

end NUMINAMATH_GPT_problem_area_triangle_PNT_l1167_116753


namespace NUMINAMATH_GPT_value_of_question_l1167_116791

noncomputable def value_of_approx : ℝ := 0.2127541038062284

theorem value_of_question :
  ((0.76^3 - 0.1^3) / (0.76^2) + value_of_approx + 0.1^2) = 0.66 :=
by
  sorry

end NUMINAMATH_GPT_value_of_question_l1167_116791


namespace NUMINAMATH_GPT_distinct_sequences_six_sided_die_rolled_six_times_l1167_116700

theorem distinct_sequences_six_sided_die_rolled_six_times :
  let count := 6
  (count ^ 6 = 46656) :=
by
  let count := 6
  sorry

end NUMINAMATH_GPT_distinct_sequences_six_sided_die_rolled_six_times_l1167_116700


namespace NUMINAMATH_GPT_john_total_distance_l1167_116716

theorem john_total_distance (speed1 time1 speed2 time2 : ℕ) (distance1 distance2 : ℕ) :
  speed1 = 35 →
  time1 = 2 →
  speed2 = 55 →
  time2 = 3 →
  distance1 = speed1 * time1 →
  distance2 = speed2 * time2 →
  distance1 + distance2 = 235 := by
  intros
  sorry

end NUMINAMATH_GPT_john_total_distance_l1167_116716


namespace NUMINAMATH_GPT_solve_for_x_l1167_116776

theorem solve_for_x (x : ℝ) : 3^(4 * x) = (81 : ℝ)^(1 / 4) → x = 1 / 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_solve_for_x_l1167_116776


namespace NUMINAMATH_GPT_total_letters_correct_l1167_116774

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end NUMINAMATH_GPT_total_letters_correct_l1167_116774


namespace NUMINAMATH_GPT_equation_of_line_passing_through_and_parallel_l1167_116741

theorem equation_of_line_passing_through_and_parallel :
  ∀ (x y : ℝ), (x = -3 ∧ y = -1) → (∃ (C : ℝ), x - 2 * y + C = 0) → C = 1 :=
by
  intros x y h₁ h₂
  sorry

end NUMINAMATH_GPT_equation_of_line_passing_through_and_parallel_l1167_116741


namespace NUMINAMATH_GPT_cost_per_load_is_25_cents_l1167_116711

-- Define the given conditions
def loads_per_bottle : ℕ := 80
def usual_price_per_bottle : ℕ := 2500 -- in cents
def sale_price_per_bottle : ℕ := 2000 -- in cents
def bottles_bought : ℕ := 2

-- Defining the total cost and total loads
def total_cost : ℕ := bottles_bought * sale_price_per_bottle
def total_loads : ℕ := bottles_bought * loads_per_bottle

-- Define the cost per load in cents
def cost_per_load_in_cents : ℕ := (total_cost * 100) / total_loads

-- Formal proof statement
theorem cost_per_load_is_25_cents 
    (h1 : loads_per_bottle = 80)
    (h2 : usual_price_per_bottle = 2500)
    (h3 : sale_price_per_bottle = 2000)
    (h4 : bottles_bought = 2)
    (h5 : total_cost = bottles_bought * sale_price_per_bottle)
    (h6 : total_loads = bottles_bought * loads_per_bottle)
    (h7 : cost_per_load_in_cents = (total_cost * 100) / total_loads):
  cost_per_load_in_cents = 25 := by
  sorry

end NUMINAMATH_GPT_cost_per_load_is_25_cents_l1167_116711


namespace NUMINAMATH_GPT_problem_statement_l1167_116710

theorem problem_statement (p q m n : ℕ) (x : ℚ)
  (h1 : p / q = 4 / 5) (h2 : m / n = 4 / 5) (h3 : x = 1 / 7) :
  x + (2 * q - p + 3 * m - 2 * n) / (2 * q + p - m + n) = 71 / 105 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1167_116710


namespace NUMINAMATH_GPT_distinct_sequences_count_l1167_116720

-- Defining the set of letters in "PROBLEMS"
def letters : List Char := ['P', 'R', 'O', 'B', 'L', 'E', 'M']

-- Defining a sequence constraint: must start with 'S' and not end with 'M'
def valid_sequence (seq : List Char) : Prop :=
  seq.head? = some 'S' ∧ seq.getLast? ≠ some 'M'

-- Counting valid sequences according to the constraints
noncomputable def count_valid_sequences : Nat :=
  6 * 120

theorem distinct_sequences_count :
  count_valid_sequences = 720 := by
  sorry

end NUMINAMATH_GPT_distinct_sequences_count_l1167_116720


namespace NUMINAMATH_GPT_fault_line_total_movement_l1167_116734

theorem fault_line_total_movement (a b : ℝ) (h1 : a = 1.25) (h2 : b = 5.25) : a + b = 6.50 := by
  -- Definitions:
  rw [h1, h2]
  -- Proof:
  sorry

end NUMINAMATH_GPT_fault_line_total_movement_l1167_116734


namespace NUMINAMATH_GPT_triangle_sine_ratio_l1167_116740

-- Define points A and C
def A : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the condition of point B being on the ellipse
def isOnEllipse (B : ℝ × ℝ) : Prop :=
  (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1

-- Define the sin law ratio we need to prove
noncomputable def sin_ratio (sin_A sin_C sin_B : ℝ) : ℝ := 
  (sin_A + sin_C) / sin_B

-- Prove the required sine ratio condition
theorem triangle_sine_ratio (B : ℝ × ℝ) (sin_A sin_C sin_B : ℝ)
  (hB : isOnEllipse B) (hA : sin_A = 0) (hC : sin_C = 0) (hB_nonzero : sin_B ≠ 0) :
  sin_ratio sin_A sin_C sin_B = 2 :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_triangle_sine_ratio_l1167_116740


namespace NUMINAMATH_GPT_amazing_squares_exist_l1167_116743

structure Quadrilateral :=
(A B C D : Point)

def diagonals_not_perpendicular (quad : Quadrilateral) : Prop := sorry -- The precise definition will abstractly represent the non-perpendicularity of diagonals.

def amazing_square (quad : Quadrilateral) (square : Square) : Prop :=
  -- Definition stating that the sides of the square (extended if necessary) pass through distinct vertices of the quadrilateral
  sorry

theorem amazing_squares_exist (quad : Quadrilateral) (h : diagonals_not_perpendicular quad) :
  ∃ squares : Finset Square, squares.card ≥ 6 ∧ ∀ square ∈ squares, amazing_square quad square :=
by sorry

end NUMINAMATH_GPT_amazing_squares_exist_l1167_116743


namespace NUMINAMATH_GPT_number_of_solutions_l1167_116755

theorem number_of_solutions (h₁ : ∀ x, 50 * x % 100 = 0 → (x % 2 = 0)) 
                            (h₂ : ∀ x, (x % 2 = 0) → (∀ k, 1 ≤ k ∧ k ≤ 49 → (k * x % 100 ≠ 0)))
                            (h₃ : ∀ x, 1 ≤ x ∧ x ≤ 100) : 
  ∃ count, count = 20 := 
by {
  -- Here, we usually would provide a method to count all valid x values meeting the conditions,
  -- but we skip the proof as instructed.
  sorry
}

end NUMINAMATH_GPT_number_of_solutions_l1167_116755


namespace NUMINAMATH_GPT_domain_of_f_l1167_116797

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (Real.log x / Real.log 2 - 1))

theorem domain_of_f :
  {x : ℝ | x > 2} = {x : ℝ | x > 0 ∧ Real.log x / Real.log 2 - 1 > 0} := 
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1167_116797


namespace NUMINAMATH_GPT_john_bought_metres_l1167_116721

-- Define the conditions
def total_cost := 425.50
def cost_per_metre := 46.00

-- State the theorem
theorem john_bought_metres : total_cost / cost_per_metre = 9.25 :=
by
  sorry

end NUMINAMATH_GPT_john_bought_metres_l1167_116721


namespace NUMINAMATH_GPT_a4_equals_zero_l1167_116781

-- Define the general term of the sequence
def a (n : ℕ) (h : n > 0) : ℤ := n^2 - 3 * n - 4

-- The theorem statement to prove a_4 = 0
theorem a4_equals_zero : a 4 (by norm_num) = 0 :=
sorry

end NUMINAMATH_GPT_a4_equals_zero_l1167_116781


namespace NUMINAMATH_GPT_darwin_spending_fraction_l1167_116771

theorem darwin_spending_fraction {x : ℝ} (h1 : 600 - 600 * x - (1 / 4) * (600 - 600 * x) = 300) :
  x = 1 / 3 :=
sorry

end NUMINAMATH_GPT_darwin_spending_fraction_l1167_116771


namespace NUMINAMATH_GPT_train_passing_time_l1167_116785

theorem train_passing_time
  (length_of_train : ℝ)
  (speed_in_kmph : ℝ)
  (conversion_factor : ℝ)
  (speed_in_mps : ℝ)
  (time : ℝ)
  (H1 : length_of_train = 65)
  (H2 : speed_in_kmph = 36)
  (H3 : conversion_factor = 5 / 18)
  (H4 : speed_in_mps = speed_in_kmph * conversion_factor)
  (H5 : time = length_of_train / speed_in_mps) :
  time = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_train_passing_time_l1167_116785


namespace NUMINAMATH_GPT_original_planned_length_l1167_116799

theorem original_planned_length (x : ℝ) (h1 : x > 0) (total_length : ℝ := 3600) (efficiency_ratio : ℝ := 1.8) (time_saved : ℝ := 20) 
  (h2 : total_length / x - total_length / (efficiency_ratio * x) = time_saved) :
  x = 80 :=
sorry

end NUMINAMATH_GPT_original_planned_length_l1167_116799


namespace NUMINAMATH_GPT_exists_square_with_only_invisible_points_l1167_116767

def is_invisible (p q : ℤ) : Prop := Int.gcd p q > 1

def all_points_in_square_invisible (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 2 ∧ ∀ x y : ℕ, (x < n ∧ y < n) → is_invisible (k*x) (k*y)

theorem exists_square_with_only_invisible_points (n : ℕ) :
  all_points_in_square_invisible n := sorry

end NUMINAMATH_GPT_exists_square_with_only_invisible_points_l1167_116767


namespace NUMINAMATH_GPT_area_of_rhombus_l1167_116764

-- Defining the lengths of the diagonals
variable (d1 d2 : ℝ)
variable (d1_eq : d1 = 15)
variable (d2_eq : d2 = 20)

-- Goal is to prove the area given the diagonal lengths
theorem area_of_rhombus (d1 d2 : ℝ) (d1_eq : d1 = 15) (d2_eq : d2 = 20) : 
  (d1 * d2) / 2 = 150 := 
by
  -- Using the given conditions for the proof
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l1167_116764


namespace NUMINAMATH_GPT_movie_final_length_l1167_116779

theorem movie_final_length (original_length : ℕ) (cut_length : ℕ) (final_length : ℕ) 
  (h1 : original_length = 60) (h2 : cut_length = 8) : 
  final_length = 52 :=
by
  sorry

end NUMINAMATH_GPT_movie_final_length_l1167_116779


namespace NUMINAMATH_GPT_sum_of_squares_of_four_integers_equals_175_l1167_116780

theorem sum_of_squares_of_four_integers_equals_175 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a^2 + b^2 + c^2 + d^2 = 175 ∧ a + b + c + d = 23 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_four_integers_equals_175_l1167_116780


namespace NUMINAMATH_GPT_inequality_not_always_hold_l1167_116796

theorem inequality_not_always_hold (a b : ℝ) (h : a > -b) : ¬ (∀ a b : ℝ, a > -b → (1 / a + 1 / b > 0)) :=
by
  intro h2
  have h3 := h2 a b h
  sorry

end NUMINAMATH_GPT_inequality_not_always_hold_l1167_116796


namespace NUMINAMATH_GPT_unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l1167_116751

noncomputable def f (x k : ℝ) : ℝ := (Real.log x) - k * x + k

theorem unique_solution_f_geq_0 {k : ℝ} :
  (∃! x : ℝ, 0 < x ∧ f x k ≥ 0) ↔ k = 1 :=
sorry

theorem inequality_hold_for_a_leq_1 {a x : ℝ} (h₀ : a ≤ 1) :
  x * (f x 1 + x - 1) < Real.exp x - a * x^2 - 1 :=
sorry

end NUMINAMATH_GPT_unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l1167_116751


namespace NUMINAMATH_GPT_find_f_ln2_l1167_116759

noncomputable def f : ℝ → ℝ := sorry

axiom fx_monotonic : Monotone f
axiom fx_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - Real.exp 1

theorem find_f_ln2 : f (Real.log 2) = -1 := 
sorry

end NUMINAMATH_GPT_find_f_ln2_l1167_116759


namespace NUMINAMATH_GPT_expression_values_l1167_116729

theorem expression_values (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ b)
  (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 := 
sorry

end NUMINAMATH_GPT_expression_values_l1167_116729


namespace NUMINAMATH_GPT_smallest_w_l1167_116730

theorem smallest_w (w : ℕ) (w_pos : w > 0) (h1 : ∀ n : ℕ, 2^4 ∣ 1452 * w)
                              (h2 : ∀ n : ℕ, 3^3 ∣ 1452 * w)
                              (h3 : ∀ n : ℕ, 13^3 ∣ 1452 * w) :
  w = 676 := sorry

end NUMINAMATH_GPT_smallest_w_l1167_116730


namespace NUMINAMATH_GPT_each_interior_angle_of_regular_octagon_l1167_116702

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end NUMINAMATH_GPT_each_interior_angle_of_regular_octagon_l1167_116702


namespace NUMINAMATH_GPT_system_of_equations_solution_l1167_116706

theorem system_of_equations_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 3) : 
  x = 4 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1167_116706


namespace NUMINAMATH_GPT_cream_cheese_cost_l1167_116717

theorem cream_cheese_cost:
  ∃ (B C : ℝ), (2 * B + 3 * C = 12) ∧ (4 * B + 2 * C = 14) ∧ (C = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_cream_cheese_cost_l1167_116717


namespace NUMINAMATH_GPT_maximize_distance_l1167_116765

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end NUMINAMATH_GPT_maximize_distance_l1167_116765


namespace NUMINAMATH_GPT_baskets_weight_l1167_116793

theorem baskets_weight 
  (weight_per_basket : ℕ)
  (num_baskets : ℕ)
  (total_weight : ℕ) 
  (h1 : weight_per_basket = 30)
  (h2 : num_baskets = 8)
  (h3 : total_weight = weight_per_basket * num_baskets) :
  total_weight = 240 := 
by
  sorry

end NUMINAMATH_GPT_baskets_weight_l1167_116793


namespace NUMINAMATH_GPT_exponential_function_solution_l1167_116709

theorem exponential_function_solution (a : ℝ) (h₁ : ∀ x : ℝ, a ^ x > 0) :
  (∃ y : ℝ, y = a ^ 2 ∧ y = 4) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_solution_l1167_116709


namespace NUMINAMATH_GPT_least_number_leaving_remainder_4_l1167_116754

theorem least_number_leaving_remainder_4 (x : ℤ) : 
  (x % 6 = 4) ∧ (x % 9 = 4) ∧ (x % 12 = 4) ∧ (x % 18 = 4) → x = 40 :=
by
  sorry

end NUMINAMATH_GPT_least_number_leaving_remainder_4_l1167_116754


namespace NUMINAMATH_GPT_total_capacity_is_1600_l1167_116733

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end NUMINAMATH_GPT_total_capacity_is_1600_l1167_116733


namespace NUMINAMATH_GPT_clock_angle_at_3_40_l1167_116750

theorem clock_angle_at_3_40
  (hour_position : ℕ → ℝ)
  (minute_position : ℕ → ℝ)
  (h_hour : hour_position 3 = 3 * 30)
  (h_minute : minute_position 40 = 40 * 6)
  : abs (minute_position 40 - (hour_position 3 + 20 * 30 / 60)) = 130 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_clock_angle_at_3_40_l1167_116750


namespace NUMINAMATH_GPT_express_recurring_decimal_as_fraction_l1167_116777

theorem express_recurring_decimal_as_fraction (h : 0.01 = (1 : ℚ) / 99) : 2.02 = (200 : ℚ) / 99 :=
by 
  sorry

end NUMINAMATH_GPT_express_recurring_decimal_as_fraction_l1167_116777


namespace NUMINAMATH_GPT_simplify_expression_l1167_116758

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 6) * (2 * x + 8) - (x + 6) * (3 * x + 1) = 3 * x^2 - 7 * x - 54 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1167_116758


namespace NUMINAMATH_GPT_cost_price_of_book_l1167_116707

theorem cost_price_of_book
(marked_price : ℝ)
(list_price : ℝ)
(cost_price : ℝ)
(h1 : marked_price = 69.85)
(h2 : list_price = marked_price * 0.85)
(h3 : list_price = cost_price * 1.25) :
cost_price = 65.75 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_book_l1167_116707


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1167_116768

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a3 : a 3 = -1)
  (h_a7 : a 7 = -9) : a 5 = -3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1167_116768


namespace NUMINAMATH_GPT_variance_of_data_set_l1167_116704

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_data_set_l1167_116704


namespace NUMINAMATH_GPT_defense_attorney_mistake_l1167_116701

variable (P Q : Prop)

theorem defense_attorney_mistake (h1 : P → Q) (h2 : ¬ (P → Q)) : P ∧ ¬ Q :=
by {
  sorry
}

end NUMINAMATH_GPT_defense_attorney_mistake_l1167_116701


namespace NUMINAMATH_GPT_number_of_uncracked_seashells_l1167_116725

theorem number_of_uncracked_seashells (toms_seashells freds_seashells cracked_seashells : ℕ) 
  (h_tom : toms_seashells = 15) 
  (h_fred : freds_seashells = 43) 
  (h_cracked : cracked_seashells = 29) : 
  toms_seashells + freds_seashells - cracked_seashells = 29 :=
by
  sorry

end NUMINAMATH_GPT_number_of_uncracked_seashells_l1167_116725


namespace NUMINAMATH_GPT_even_function_a_value_l1167_116727

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (a * (-x)^2 + (2 * a + 1) * (-x) - 1) = (a * x^2 + (2 * a + 1) * x - 1)) →
  a = - 1 / 2 :=
by sorry

end NUMINAMATH_GPT_even_function_a_value_l1167_116727


namespace NUMINAMATH_GPT_books_in_library_l1167_116728

theorem books_in_library (n_shelves : ℕ) (n_books_per_shelf : ℕ) (h_shelves : n_shelves = 1780) (h_books_per_shelf : n_books_per_shelf = 8) :
  n_shelves * n_books_per_shelf = 14240 :=
by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_books_in_library_l1167_116728


namespace NUMINAMATH_GPT_trigonometric_identity_l1167_116748

theorem trigonometric_identity (α : Real) (h : (1 + Real.sin α) / Real.cos α = -1 / 2) :
  (Real.cos α) / (Real.sin α - 1) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1167_116748


namespace NUMINAMATH_GPT_max_smoothie_servings_l1167_116766

def servings (bananas yogurt strawberries : ℕ) : ℕ :=
  min (bananas * 4 / 3) (min (yogurt * 4 / 2) (strawberries * 4 / 1))

theorem max_smoothie_servings :
  servings 9 10 3 = 12 :=
by
  -- Proof steps would be inserted here
  sorry

end NUMINAMATH_GPT_max_smoothie_servings_l1167_116766


namespace NUMINAMATH_GPT_leak_emptying_time_l1167_116722

-- Definitions based on given conditions
def tank_fill_rate_without_leak : ℚ := 1 / 3
def combined_fill_and_leak_rate : ℚ := 1 / 4

-- Leak emptying time to be proven
theorem leak_emptying_time (R : ℚ := tank_fill_rate_without_leak) (C : ℚ := combined_fill_and_leak_rate) :
  (1 : ℚ) / (R - C) = 12 := by
  sorry

end NUMINAMATH_GPT_leak_emptying_time_l1167_116722


namespace NUMINAMATH_GPT_total_yards_run_l1167_116731

-- Define the yardages and games for each athlete
def Malik_yards_per_game : ℕ := 18
def Malik_games : ℕ := 5

def Josiah_yards_per_game : ℕ := 22
def Josiah_games : ℕ := 7

def Darnell_yards_per_game : ℕ := 11
def Darnell_games : ℕ := 4

def Kade_yards_per_game : ℕ := 15
def Kade_games : ℕ := 6

-- Prove that the total yards run by the four athletes is 378
theorem total_yards_run :
  (Malik_yards_per_game * Malik_games) +
  (Josiah_yards_per_game * Josiah_games) +
  (Darnell_yards_per_game * Darnell_games) +
  (Kade_yards_per_game * Kade_games) = 378 :=
by
  sorry

end NUMINAMATH_GPT_total_yards_run_l1167_116731


namespace NUMINAMATH_GPT_cody_games_still_has_l1167_116746

def initial_games : ℕ := 9
def games_given_away_to_jake : ℕ := 4
def games_given_away_to_sarah : ℕ := 2
def games_bought_over_weekend : ℕ := 3

theorem cody_games_still_has : 
  initial_games - (games_given_away_to_jake + games_given_away_to_sarah) + games_bought_over_weekend = 6 := 
by
  sorry

end NUMINAMATH_GPT_cody_games_still_has_l1167_116746


namespace NUMINAMATH_GPT_discount_is_5_percent_l1167_116749

-- Defining the conditions
def cost_per_iphone : ℕ := 600
def total_cost_3_iphones : ℕ := 3 * cost_per_iphone
def savings : ℕ := 90

-- Calculating the discount percentage
def discount_percentage : ℕ := (savings * 100) / total_cost_3_iphones

-- Stating the theorem
theorem discount_is_5_percent : discount_percentage = 5 :=
  sorry

end NUMINAMATH_GPT_discount_is_5_percent_l1167_116749


namespace NUMINAMATH_GPT_donut_distribution_l1167_116789

theorem donut_distribution :
  ∃ (Alpha Beta Gamma Delta Epsilon : ℕ), 
    Delta = 8 ∧ 
    Beta = 3 * Gamma ∧ 
    Alpha = 2 * Delta ∧ 
    Epsilon = Gamma - 4 ∧ 
    Alpha + Beta + Gamma + Delta + Epsilon = 60 ∧ 
    Alpha = 16 ∧ 
    Beta = 24 ∧ 
    Gamma = 8 ∧ 
    Delta = 8 ∧ 
    Epsilon = 4 :=
by
  sorry

end NUMINAMATH_GPT_donut_distribution_l1167_116789


namespace NUMINAMATH_GPT_area_of_circle_eq_sixteen_pi_l1167_116742

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_eq_sixteen_pi_l1167_116742


namespace NUMINAMATH_GPT_vacant_seats_l1167_116713

open Nat

-- Define the conditions as Lean definitions
def num_tables : Nat := 5
def seats_per_table : Nat := 8
def occupied_tables : Nat := 2
def people_per_occupied_table : Nat := 3
def unusable_tables : Nat := 1

-- Calculate usable tables
def usable_tables : Nat := num_tables - unusable_tables

-- Calculate total occupied people
def total_occupied_people : Nat := occupied_tables * people_per_occupied_table

-- Calculate total seats for occupied tables
def total_seats_occupied_tables : Nat := occupied_tables * seats_per_table

-- Calculate vacant seats in occupied tables
def vacant_seats_occupied_tables : Nat := total_seats_occupied_tables - total_occupied_people

-- Calculate completely unoccupied tables
def unoccupied_tables : Nat := usable_tables - occupied_tables

-- Calculate total seats for unoccupied tables
def total_seats_unoccupied_tables : Nat := unoccupied_tables * seats_per_table

-- Calculate total vacant seats
def total_vacant_seats : Nat := vacant_seats_occupied_tables + total_seats_unoccupied_tables

-- Theorem statement to prove
theorem vacant_seats : total_vacant_seats = 26 := by
  sorry

end NUMINAMATH_GPT_vacant_seats_l1167_116713


namespace NUMINAMATH_GPT_sofia_total_time_l1167_116724

-- Definitions for the conditions
def laps : ℕ := 5
def track_length : ℕ := 400  -- in meters
def speed_first_100 : ℕ := 4  -- meters per second
def speed_remaining_300 : ℕ := 5  -- meters per second

-- Times taken for respective distances
def time_first_100 (distance speed : ℕ) : ℕ := distance / speed
def time_remaining_300 (distance speed : ℕ) : ℕ := distance / speed

def time_one_lap : ℕ := time_first_100 100 speed_first_100 + time_remaining_300 300 speed_remaining_300
def total_time_seconds : ℕ := laps * time_one_lap
def total_time_minutes : ℕ := 7
def total_time_extra_seconds : ℕ := 5

-- Problem statement
theorem sofia_total_time :
  total_time_seconds = total_time_minutes * 60 + total_time_extra_seconds :=
by
  sorry

end NUMINAMATH_GPT_sofia_total_time_l1167_116724


namespace NUMINAMATH_GPT_sum_geometric_sequence_l1167_116786

theorem sum_geometric_sequence {n : ℕ} (S : ℕ → ℝ) (h1 : S n = 10) (h2 : S (2 * n) = 30) : 
  S (3 * n) = 70 := 
by 
  sorry

end NUMINAMATH_GPT_sum_geometric_sequence_l1167_116786


namespace NUMINAMATH_GPT_equation_solutions_l1167_116738

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃ x : ℝ, ax + b = 0) ∨ (∃ x : ℝ, ∀ y : ℝ, ax + b = 0 → x = y) :=
sorry

end NUMINAMATH_GPT_equation_solutions_l1167_116738


namespace NUMINAMATH_GPT_FO_greater_DI_l1167_116798

-- The quadrilateral FIDO is assumed to be convex with specified properties
variables {F I D O E : Type*}

variables (length_FI length_DI length_DO length_FO : ℝ)
variables (angle_FIO angle_DIO : ℝ)
variables (E : I)

-- Given conditions
variables (convex_FIDO : Prop) -- FIDO is convex
variables (h1 : length_FI = length_DO)
variables (h2 : length_FI > length_DI)
variables (h3 : angle_FIO = angle_DIO)

-- Use given identity IE = ID
variables (length_IE : ℝ) (h4 : length_IE = length_DI)

theorem FO_greater_DI 
    (length_FI length_DI length_DO length_FO : ℝ)
    (angle_FIO angle_DIO : ℝ)
    (convex_FIDO : Prop)
    (h1 : length_FI = length_DO)
    (h2 : length_FI > length_DI)
    (h3 : angle_FIO = angle_DIO)
    (length_IE : ℝ)
    (h4 : length_IE = length_DI) : 
    length_FO > length_DI :=
sorry

end NUMINAMATH_GPT_FO_greater_DI_l1167_116798


namespace NUMINAMATH_GPT_worker_and_robot_capacity_additional_workers_needed_l1167_116737

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end NUMINAMATH_GPT_worker_and_robot_capacity_additional_workers_needed_l1167_116737


namespace NUMINAMATH_GPT_polynomial_expansion_sum_eq_l1167_116778

theorem polynomial_expansion_sum_eq :
  (∀ (x : ℝ), (2 * x - 1)^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 243) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_sum_eq_l1167_116778


namespace NUMINAMATH_GPT_total_right_handed_players_is_60_l1167_116712

def total_players : ℕ := 70
def throwers : ℕ := 40
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed_players : ℕ := right_handed_throwers + right_handed_non_throwers

theorem total_right_handed_players_is_60 : total_right_handed_players = 60 := by
  sorry

end NUMINAMATH_GPT_total_right_handed_players_is_60_l1167_116712


namespace NUMINAMATH_GPT_unit_vector_parallel_to_a_l1167_116792

theorem unit_vector_parallel_to_a (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 12 * y = 5 * x) :
  (x = 12 / 13 ∧ y = 5 / 13) ∨ (x = -12 / 13 ∧ y = -5 / 13) := by
  sorry

end NUMINAMATH_GPT_unit_vector_parallel_to_a_l1167_116792


namespace NUMINAMATH_GPT_gecko_sales_ratio_l1167_116769

theorem gecko_sales_ratio (x : ℕ) (h1 : 86 + x = 258) : 86 / Nat.gcd 172 86 = 1 ∧ 172 / Nat.gcd 172 86 = 2 := by
  sorry

end NUMINAMATH_GPT_gecko_sales_ratio_l1167_116769


namespace NUMINAMATH_GPT_phil_quarters_collection_l1167_116787

theorem phil_quarters_collection
    (initial_quarters : ℕ)
    (doubled_quarters : ℕ)
    (additional_quarters_per_month : ℕ)
    (total_quarters_end_of_second_year : ℕ)
    (quarters_collected_every_third_month : ℕ)
    (total_quarters_end_of_third_year : ℕ)
    (remaining_quarters_after_loss : ℕ)
    (quarters_left : ℕ) :
    initial_quarters = 50 →
    doubled_quarters = 2 * initial_quarters →
    additional_quarters_per_month = 3 →
    total_quarters_end_of_second_year = doubled_quarters + 12 * additional_quarters_per_month →
    total_quarters_end_of_third_year = total_quarters_end_of_second_year + 4 * quarters_collected_every_third_month →
    remaining_quarters_after_loss = (3 / 4 : ℚ) * total_quarters_end_of_third_year → 
    quarters_left = 105 →
    quarters_collected_every_third_month = 1 := 
by
  sorry

end NUMINAMATH_GPT_phil_quarters_collection_l1167_116787


namespace NUMINAMATH_GPT_proof_problem_l1167_116735

structure Plane := (name : String)
structure Line := (name : String)

def parallel_planes (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def parallel_lines (m n : Line) : Prop := sorry

theorem proof_problem (m : Line) (α β : Plane) :
  parallel_planes α β → in_plane m α → parallel_lines m (Line.mk β.name) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1167_116735


namespace NUMINAMATH_GPT_snail_crawl_distance_l1167_116782

theorem snail_crawl_distance
  (α : ℕ → ℝ)  -- α represents the snail's position at each minute
  (crawls_forward : ∀ n m : ℕ, n < m → α n ≤ α m)  -- The snail moves forward (without going backward)
  (observer_finds : ∀ n : ℕ, α (n + 1) - α n = 1) -- Every observer finds that the snail crawled exactly 1 meter per minute
  (time_span : ℕ := 6)  -- Total observation period is 6 minutes
  : α time_span - α 0 ≤ 10 :=  -- The distance crawled in 6 minutes does not exceed 10 meters
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_snail_crawl_distance_l1167_116782


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l1167_116708

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b)) → a = b :=
sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l1167_116708


namespace NUMINAMATH_GPT_scientific_notation_example_l1167_116719

theorem scientific_notation_example :
  ∃ (a : ℝ) (b : ℤ), 1300000 = a * 10 ^ b ∧ a = 1.3 ∧ b = 6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_example_l1167_116719


namespace NUMINAMATH_GPT_total_lunch_bill_l1167_116794

theorem total_lunch_bill (hotdog salad : ℝ) (h1 : hotdog = 5.36) (h2 : salad = 5.10) : hotdog + salad = 10.46 := 
by
  rw [h1, h2]
  norm_num
  

end NUMINAMATH_GPT_total_lunch_bill_l1167_116794


namespace NUMINAMATH_GPT_inequality_holds_if_and_only_if_l1167_116790

noncomputable def absolute_inequality (x a : ℝ) : Prop :=
  |x - 3| + |x - 4| + |x - 5| < a

theorem inequality_holds_if_and_only_if (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, absolute_inequality x a) ↔ a > 4 := 
sorry

end NUMINAMATH_GPT_inequality_holds_if_and_only_if_l1167_116790


namespace NUMINAMATH_GPT_find_a_perpendicular_lines_l1167_116736

theorem find_a_perpendicular_lines (a : ℝ) :
  (∀ (x y : ℝ),
    a * x + 2 * y + 6 = 0 → 
    x + (a - 1) * y + a^2 - 1 = 0 → (a * 1 + 2 * (a - 1) = 0)) → 
  a = 2/3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_a_perpendicular_lines_l1167_116736


namespace NUMINAMATH_GPT_value_of_nabla_expression_l1167_116714

namespace MathProblem

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem value_of_nabla_expression : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end MathProblem

end NUMINAMATH_GPT_value_of_nabla_expression_l1167_116714


namespace NUMINAMATH_GPT_steve_and_laura_meet_time_l1167_116732

structure PathsOnParallelLines where
  steve_speed : ℝ
  laura_speed : ℝ
  path_separation : ℝ
  art_diameter : ℝ
  initial_distance_hidden : ℝ

def meet_time (p : PathsOnParallelLines) : ℝ :=
  sorry -- To be proven

-- Define the specific case for Steve and Laura
def steve_and_laura_paths : PathsOnParallelLines :=
  { steve_speed := 3,
    laura_speed := 1,
    path_separation := 240,
    art_diameter := 80,
    initial_distance_hidden := 230 }

theorem steve_and_laura_meet_time :
  meet_time steve_and_laura_paths = 45 :=
  sorry

end NUMINAMATH_GPT_steve_and_laura_meet_time_l1167_116732


namespace NUMINAMATH_GPT_ram_krish_task_completion_l1167_116739

/-!
  Given:
  1. Ram's efficiency (R) is half of Krish's efficiency (K).
  2. Ram can complete the task alone in 24 days.

  To Prove:
  Ram and Krish will complete the task together in 8 days.
-/

theorem ram_krish_task_completion {R K : ℝ} (hR : R = 1 / 2 * K)
  (hRAMalone : R ≠ 0) (hRAMtime : 24 * R = 1) :
  1 / (R + K) = 8 := by
  sorry

end NUMINAMATH_GPT_ram_krish_task_completion_l1167_116739


namespace NUMINAMATH_GPT_sphere_cone_radius_ratio_l1167_116747

-- Define the problem using given conditions and expected outcome.
theorem sphere_cone_radius_ratio (r R h : ℝ)
  (h1 : h = 2 * r)
  (h2 : (1/3) * π * R^2 * h = 3 * (4/3) * π * r^3) :
  r / R = 1 / Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_sphere_cone_radius_ratio_l1167_116747
