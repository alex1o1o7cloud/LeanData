import Mathlib

namespace eldora_boxes_paper_clips_l2420_242043

theorem eldora_boxes_paper_clips (x y : ℝ)
  (h1 : 1.85 * x + 7 * y = 55.40)
  (h2 : 1.85 * 12 + 10 * y = 61.70)
  (h3 : 1.85 = 1.85) : -- Given && Asserting the constant price of one box

  x = 15 :=
by
  sorry

end eldora_boxes_paper_clips_l2420_242043


namespace profit_is_eight_dollars_l2420_242035

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end profit_is_eight_dollars_l2420_242035


namespace sequence_inequality_l2420_242026

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 
  if n = 0 then 1/2
  else a (n - 1) + (1 / (n:ℚ)^2) * (a (n - 1))^2

theorem sequence_inequality (n : ℕ) : 
  1 - 1 / 2 ^ (n + 1) ≤ a n ∧ a n < 7 / 5 := 
sorry

end sequence_inequality_l2420_242026


namespace final_bicycle_price_l2420_242067

-- Define conditions 
def original_price : ℝ := 200
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def price_after_first_discount := original_price * (1 - first_discount)
def final_price := price_after_first_discount * (1 - second_discount)

-- Define the Lean statement to be proven
theorem final_bicycle_price :
  final_price = 120 :=
by
  -- Proof goes here
  sorry

end final_bicycle_price_l2420_242067


namespace prism_faces_l2420_242044

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end prism_faces_l2420_242044


namespace basketball_free_throws_l2420_242027

theorem basketball_free_throws (total_players : ℕ) (number_captains : ℕ) (players_not_including_one : ℕ) 
  (free_throws_per_captain : ℕ) (total_free_throws : ℕ) 
  (h1 : total_players = 15)
  (h2 : number_captains = 2)
  (h3 : players_not_including_one = total_players - 1)
  (h4 : free_throws_per_captain = players_not_including_one * number_captains)
  (h5 : total_free_throws = free_throws_per_captain)
  : total_free_throws = 28 :=
by
  -- Proof is not required, so we provide sorry to skip it.
  sorry

end basketball_free_throws_l2420_242027


namespace find_m_l2420_242077

open Set

theorem find_m (m : ℝ) (A B : Set ℝ)
  (h1 : A = {-1, 3, 2 * m - 1})
  (h2 : B = {3, m})
  (h3 : B ⊆ A) : m = 1 ∨ m = -1 :=
by
  sorry

end find_m_l2420_242077


namespace exactly_one_even_l2420_242054

theorem exactly_one_even (a b c : ℕ) : 
  (∀ x, ¬ (a = x ∧ b = x ∧ c = x) ∧ 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ b % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ c % 2 = 0) ∧ 
  ¬ (b % 2 = 0 ∧ c % 2 = 0)) :=
by
  sorry

end exactly_one_even_l2420_242054


namespace stacy_days_to_finish_l2420_242070

-- Definitions based on the conditions
def total_pages : ℕ := 81
def pages_per_day : ℕ := 27

-- The theorem statement
theorem stacy_days_to_finish : total_pages / pages_per_day = 3 := by
  -- the proof is omitted
  sorry

end stacy_days_to_finish_l2420_242070


namespace valid_t_range_for_f_l2420_242076

theorem valid_t_range_for_f :
  (∀ x : ℝ, |x + 1| + |x - t| ≥ 2015) ↔ t ∈ (Set.Iic (-2016) ∪ Set.Ici 2014) := 
sorry

end valid_t_range_for_f_l2420_242076


namespace sum_of_consecutive_integers_420_l2420_242068

theorem sum_of_consecutive_integers_420 : 
  ∃ (k n : ℕ) (h1 : k ≥ 2) (h2 : k * n + k * (k - 1) / 2 = 420), 
  ∃ K : Finset ℕ, K.card = 6 ∧ (∀ x ∈ K, k = x) :=
by
  sorry

end sum_of_consecutive_integers_420_l2420_242068


namespace solve_equation_l2420_242053

theorem solve_equation (x : ℝ) (h : -x^2 = (3 * x + 1) / (x + 3)) : x = -1 :=
sorry

end solve_equation_l2420_242053


namespace Carter_gave_Marcus_58_cards_l2420_242038

-- Define the conditions as variables
def original_cards : ℕ := 210
def current_cards : ℕ := 268

-- Define the question as a function
def cards_given_by_carter (original current : ℕ) : ℕ := current - original

-- Statement that we need to prove
theorem Carter_gave_Marcus_58_cards : cards_given_by_carter original_cards current_cards = 58 :=
by
  -- Proof goes here
  sorry

end Carter_gave_Marcus_58_cards_l2420_242038


namespace correct_word_is_any_l2420_242045

def words : List String := ["other", "any", "none", "some"]

def is_correct_word (word : String) : Prop :=
  "Jane was asked a lot of questions, but she didn’t answer " ++ word ++ " of them." = 
    "Jane was asked a lot of questions, but she didn’t answer any of them."

theorem correct_word_is_any : is_correct_word "any" :=
by
  sorry

end correct_word_is_any_l2420_242045


namespace saved_per_bagel_l2420_242007

-- Definitions of the conditions
def bagel_cost_each : ℝ := 3.50
def dozen_cost : ℝ := 38
def bakers_dozen : ℕ := 13
def discount : ℝ := 0.05

-- The conjecture we need to prove
theorem saved_per_bagel : 
  let total_cost_without_discount := dozen_cost + bagel_cost_each
  let discount_amount := discount * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount_amount
  let cost_per_bagel_without_discount := dozen_cost / 12
  let cost_per_bagel_with_discount := total_cost_with_discount / bakers_dozen
  let savings_per_bagel := cost_per_bagel_without_discount - cost_per_bagel_with_discount
  let savings_in_cents := savings_per_bagel * 100
  savings_in_cents = 13.36 :=
by
  -- Placeholder for the actual proof
  sorry

end saved_per_bagel_l2420_242007


namespace minimum_value_of_expression_l2420_242083

noncomputable def expr (a b c : ℝ) : ℝ := 8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c)

theorem minimum_value_of_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  expr a b c ≥ 18 * Real.sqrt 3 := 
by
  sorry

end minimum_value_of_expression_l2420_242083


namespace age_difference_l2420_242074

theorem age_difference (p f : ℕ) (hp : p = 11) (hf : f = 42) : f - p = 31 :=
by
  sorry

end age_difference_l2420_242074


namespace range_of_a_l2420_242097

theorem range_of_a (a : ℝ) :
  (∃ (M : ℝ × ℝ), (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧
    (M.1)^2 + (M.2 - 2)^2 + (M.1)^2 + (M.2)^2 = 10) → 
  0 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l2420_242097


namespace average_age_increase_by_one_l2420_242057

-- Definitions based on the conditions.
def initial_average_age : ℕ := 14
def initial_students : ℕ := 10
def new_students_average_age : ℕ := 17
def new_students : ℕ := 5

-- Helper calculation for the total age of initial students.
def total_age_initial_students := initial_students * initial_average_age

-- Helper calculation for the total age of new students.
def total_age_new_students := new_students * new_students_average_age

-- Helper calculation for the total age of all students.
def total_age_all_students := total_age_initial_students + total_age_new_students

-- Helper calculation for the number of all students.
def total_students := initial_students + new_students

-- Calculate the new average age.
def new_average_age := total_age_all_students / total_students

-- The goal is to prove the increase in average age is 1 year.
theorem average_age_increase_by_one :
  new_average_age - initial_average_age = 1 :=
by
  -- Proof goes here
  sorry

end average_age_increase_by_one_l2420_242057


namespace find_A_from_conditions_l2420_242031

variable (A B C D : ℕ)
variable (h_distinct : A ≠ B) (h_distinct2 : C ≠ D)
variable (h_positive : A > 0) (h_positive2 : B > 0) (h_positive3 : C > 0) (h_positive4 : D > 0)
variable (h_product1 : A * B = 72)
variable (h_product2 : C * D = 72)
variable (h_condition : A - B = C * D)

theorem find_A_from_conditions :
  A = 3 :=
sorry

end find_A_from_conditions_l2420_242031


namespace calculate_seedlings_l2420_242093

-- Define conditions
def condition_1 (x n : ℕ) : Prop :=
  x = 5 * n + 6

def condition_2 (x m : ℕ) : Prop :=
  x = 6 * m - 9

-- Define the main theorem based on these conditions
theorem calculate_seedlings (x : ℕ) : (∃ n, condition_1 x n) ∧ (∃ m, condition_2 x m) → x = 81 :=
by {
  sorry
}

end calculate_seedlings_l2420_242093


namespace equilateral_triangle_l2420_242073

namespace TriangleEquilateral

-- Define the structure of a triangle and given conditions
structure Triangle :=
  (A B C : ℝ)  -- vertices
  (angleA : ℝ) -- angle at vertex A
  (sideBC : ℝ) -- length of side BC
  (perimeter : ℝ)  -- perimeter of the triangle

-- Define the proof problem
theorem equilateral_triangle (T : Triangle) (h1 : T.angleA = 60)
  (h2 : T.sideBC = T.perimeter / 3) : 
  T.A = T.B ∧ T.B = T.C ∧ T.A = T.C ∧ T.A = T.B ∧ T.B = T.C ∧ T.A = T.C :=
  sorry

end TriangleEquilateral

end equilateral_triangle_l2420_242073


namespace minimum_area_of_cyclic_quadrilateral_l2420_242042

theorem minimum_area_of_cyclic_quadrilateral :
  ∀ (r1 r2 : ℝ), (r1 = 1) ∧ (r2 = 2) →
    ∃ (A : ℝ), A = 3 * Real.sqrt 3 ∧ 
    (∀ (q : ℝ) (circumscribed : q ≤ A),
      ∀ (p : Prop), (p = (∃ x y z w, 
        ∀ (cx : ℝ) (cy : ℝ) (cr : ℝ), 
          cr = r2 ∧ 
          (Real.sqrt ((x - cx)^2 + (y - cy)^2) = r2) ∧ 
          (Real.sqrt ((z - cx)^2 + (w - cy)^2) = r2) ∧ 
          (Real.sqrt ((x - cx)^2 + (w - cy)^2) = r1) ∧ 
          (Real.sqrt ((z - cx)^2 + (y - cy)^2) = r1)
      )) → q = A) :=
sorry

end minimum_area_of_cyclic_quadrilateral_l2420_242042


namespace necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l2420_242002

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditionally state that x > -3 is necessary for an acute angle
theorem necessary_condition_for_acute_angle (x : ℝ) :
  dot_product vector_a (vector_b x) > 0 → x > -3 := by
  sorry

-- Define the theorem for necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > -3) → (dot_product vector_a (vector_b x) > 0 ∧ x ≠ 4 / 3) := by
  sorry

end necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l2420_242002


namespace property_damage_worth_40000_l2420_242008

-- Definitions based on conditions in a)
def medical_bills : ℝ := 70000
def insurance_rate : ℝ := 0.80
def carl_payment : ℝ := 22000
def carl_rate : ℝ := 0.20

theorem property_damage_worth_40000 :
  ∃ P : ℝ, P = 40000 ∧ 
    (carl_payment = carl_rate * (P + medical_bills)) :=
by
  sorry

end property_damage_worth_40000_l2420_242008


namespace fraction_of_white_surface_area_l2420_242081

/-- A cube has edges of 4 inches and is constructed using 64 smaller cubes, each with edges of 1 inch.
Out of these smaller cubes, 56 are white and 8 are black. The 8 black cubes fully cover one face of the larger cube.
Prove that the fraction of the surface area of the larger cube that is white is 5/6. -/
theorem fraction_of_white_surface_area 
  (total_cubes : ℕ := 64)
  (white_cubes : ℕ := 56)
  (black_cubes : ℕ := 8)
  (total_surface_area : ℕ := 96)
  (black_face_area : ℕ := 16)
  (white_surface_area : ℕ := 80) :
  white_surface_area / total_surface_area = 5 / 6 :=
sorry

end fraction_of_white_surface_area_l2420_242081


namespace tan_half_angle_l2420_242080

theorem tan_half_angle (p q : ℝ) (h_cos : Real.cos p + Real.cos q = 3 / 5) (h_sin : Real.sin p + Real.sin q = 1 / 5) : Real.tan ((p + q) / 2) = 1 / 3 :=
sorry

end tan_half_angle_l2420_242080


namespace probability_heads_tails_4_tosses_l2420_242011

-- Define the probabilities of heads and tails
variables (p q : ℝ)

-- Define the conditions
def unfair_coin (p q : ℝ) : Prop :=
  p ≠ q ∧ p + q = 1 ∧ 2 * p * q = 1/2

-- Define the theorem to prove the probability of two heads and two tails
theorem probability_heads_tails_4_tosses 
  (h_unfair : unfair_coin p q) 
  : 6 * (p * q)^2 = 3 / 8 :=
by sorry

end probability_heads_tails_4_tosses_l2420_242011


namespace highest_possible_relocation_preference_l2420_242062

theorem highest_possible_relocation_preference
  (total_employees : ℕ)
  (relocated_to_X_percent : ℝ)
  (relocated_to_Y_percent : ℝ)
  (prefer_X_percent : ℝ)
  (prefer_Y_percent : ℝ)
  (htotal : total_employees = 200)
  (hrelocated_to_X_percent : relocated_to_X_percent = 0.30)
  (hrelocated_to_Y_percent : relocated_to_Y_percent = 0.70)
  (hprefer_X_percent : prefer_X_percent = 0.60)
  (hprefer_Y_percent : prefer_Y_percent = 0.40) :
  ∃ (max_relocated_with_preference : ℕ), max_relocated_with_preference = 140 :=
by
  sorry

end highest_possible_relocation_preference_l2420_242062


namespace max_rectangle_area_l2420_242069

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l2420_242069


namespace max_inscribed_triangle_area_sum_l2420_242017

noncomputable def inscribed_triangle_area (a b : ℝ) (h_a : a = 12) (h_b : b = 13) : ℝ :=
  let s := min (a / (Real.sqrt 3 / 2)) (b / (1 / 2))
  (Real.sqrt 3 / 4) * s^2

theorem max_inscribed_triangle_area_sum :
  inscribed_triangle_area 12 13 (by rfl) (by rfl) = 48 * Real.sqrt 3 - 0 :=
by
  sorry

#eval 48 + 3 + 0
-- Expected Result: 51

end max_inscribed_triangle_area_sum_l2420_242017


namespace find_initial_speed_l2420_242005

-- Definitions for the conditions
def total_distance : ℕ := 800
def time_at_initial_speed : ℕ := 6
def time_at_60_mph : ℕ := 4
def time_at_40_mph : ℕ := 2
def speed_at_60_mph : ℕ := 60
def speed_at_40_mph : ℕ := 40

-- Setting up the equation: total distance covered
def distance_covered (v : ℕ) : ℕ :=
  time_at_initial_speed * v + time_at_60_mph * speed_at_60_mph + time_at_40_mph * speed_at_40_mph

-- Proof problem statement
theorem find_initial_speed : ∃ v : ℕ, distance_covered v = total_distance ∧ v = 80 := by
  existsi 80
  simp [distance_covered, total_distance, time_at_initial_speed, speed_at_60_mph, time_at_40_mph]
  norm_num
  sorry

end find_initial_speed_l2420_242005


namespace sufficient_condition_for_product_l2420_242086

-- Given conditions
def intersects_parabola_at_two_points (x1 y1 x2 y2 : ℝ) : Prop :=
  y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 ≠ x2

def line_through_focus (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 1)

-- The theorem to prove
theorem sufficient_condition_for_product 
  (x1 y1 x2 y2 k : ℝ)
  (h1 : intersects_parabola_at_two_points x1 y1 x2 y2)
  (h2 : line_through_focus x1 y1 k)
  (h3 : line_through_focus x2 y2 k) :
  x1 * x2 = 1 :=
sorry

end sufficient_condition_for_product_l2420_242086


namespace stack_logs_total_l2420_242000

   theorem stack_logs_total (a l d : ℤ) (n : ℕ) (top_logs : ℕ) (h1 : a = 15) (h2 : l = 5) (h3 : d = -2) (h4 : n = ((l - a) / d).natAbs + 1) (h5 : top_logs = 5) : (n / 2 : ℤ) * (a + l) = 60 :=
   by
   sorry
   
end stack_logs_total_l2420_242000


namespace largest_integer_with_square_three_digits_base_7_l2420_242030

theorem largest_integer_with_square_three_digits_base_7 : 
  ∃ M : ℕ, (7^2 ≤ M^2 ∧ M^2 < 7^3) ∧ ∀ n : ℕ, (7^2 ≤ n^2 ∧ n^2 < 7^3) → n ≤ M := 
sorry

end largest_integer_with_square_three_digits_base_7_l2420_242030


namespace minimum_value_expression_l2420_242088

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

end minimum_value_expression_l2420_242088


namespace system_of_equations_solution_l2420_242084

theorem system_of_equations_solution
  (x y z : ℤ)
  (h1 : x + y + z = 12)
  (h2 : 8 * x + 5 * y + 3 * z = 60) :
  (x = 0 ∧ y = 12 ∧ z = 0) ∨
  (x = 2 ∧ y = 7 ∧ z = 3) ∨
  (x = 4 ∧ y = 2 ∧ z = 6) :=
sorry

end system_of_equations_solution_l2420_242084


namespace absolute_value_simplification_l2420_242089

theorem absolute_value_simplification (a b : ℝ) (ha : a < 0) (hb : b > 0) : |a - b| + |b - a| = -2 * a + 2 * b := 
by 
  sorry

end absolute_value_simplification_l2420_242089


namespace basketball_probability_third_shot_l2420_242020

theorem basketball_probability_third_shot
  (p1 : ℚ) (p2_given_made1 : ℚ) (p2_given_missed1 : ℚ) (p3_given_made2 : ℚ) (p3_given_missed2 : ℚ) :
  p1 = 2 / 3 → p2_given_made1 = 2 / 3 → p2_given_missed1 = 1 / 3 → p3_given_made2 = 2 / 3 → p3_given_missed2 = 2 / 3 →
  (p1 * p2_given_made1 * p3_given_made2 + p1 * p2_given_missed1 * p3_given_misseds2 + 
   (1 - p1) * p2_given_made1 * p3_given_made2 + (1 - p1) * p2_given_missed1 * p3_given_missed2) = 14 / 27 :=
by
  sorry

end basketball_probability_third_shot_l2420_242020


namespace expression_evaluates_to_one_l2420_242032

theorem expression_evaluates_to_one :
  (1 / 3)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) + (Real.pi - 2016)^0 - (8:ℝ)^(1/3) = 1 :=
by
  -- step-by-step simplification skipped, as per requirements
  sorry

end expression_evaluates_to_one_l2420_242032


namespace husband_and_wife_age_l2420_242075

theorem husband_and_wife_age (x y : ℕ) (h1 : 11 * x = 2 * (22 * y - 11 * x)) (h2 : 11 * x ≠ 0) (h3 : 11 * y ≠ 0) (h4 : 11 * (x + y) ≤ 99) : 
  x = 4 ∧ y = 3 :=
by
  sorry

end husband_and_wife_age_l2420_242075


namespace mod_squares_eq_one_l2420_242025

theorem mod_squares_eq_one
  (n : ℕ)
  (h : n = 5)
  (a : ℤ)
  (ha : ∃ b : ℕ, ↑b = a ∧ b * b ≡ 1 [MOD 5]) :
  (a * a) % n = 1 :=
by
  sorry

end mod_squares_eq_one_l2420_242025


namespace seeds_germinated_percentage_l2420_242036

theorem seeds_germinated_percentage (n1 n2 : ℕ) (p1 p2 : ℝ) (h1 : n1 = 300) (h2 : n2 = 200) (h3 : p1 = 0.25) (h4 : p2 = 0.30) :
  ( (n1 * p1 + n2 * p2) / (n1 + n2) ) * 100 = 27 :=
by
  sorry

end seeds_germinated_percentage_l2420_242036


namespace intersection_A_B_l2420_242090

namespace MathProof

open Set

def A := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2 * x + 6}

theorem intersection_A_B : A ∩ B = Icc (-1 : ℝ) 7 :=
by
  sorry

end MathProof

end intersection_A_B_l2420_242090


namespace general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l2420_242071

-- Defines the sequences and properties given in the problem
def sequences (a_n b_n S_n T_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ S_n 2 = 4 ∧ 
  (∀ n : ℕ, 3 * S_n (n + 1) = 2 * S_n n + S_n (n + 2) + a_n n)

-- (1) Prove the general formula for {a_n}
theorem general_formula_for_a_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- (2) If {b_n} is an arithmetic sequence and ∀n ∈ ℕ, S_n > T_n, prove a_n > b_n
theorem a_n_greater_than_b_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (arithmetic_b : ∃ d: ℕ, ∀ n: ℕ, b_n n = b_n 0 + n * d)
  (Sn_greater_Tn : ∀ (n : ℕ), S_n n > T_n n) :
  ∀ n : ℕ, a_n n > b_n n :=
sorry

-- (3) If {b_n} is a geometric sequence, find n such that (a_n + 2 * T_n) / (b_n + 2 * S_n) = a_k
theorem find_n_in_geometric_sequence
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (geometric_b : ∃ r: ℕ, ∀ n: ℕ, b_n n = b_n 0 * r^n)
  (b1_eq_1 : b_n 1 = 1)
  (b2_eq_3 : b_n 2 = 3)
  (k : ℕ) :
  ∃ n : ℕ, (a_n n + 2 * T_n n) / (b_n n + 2 * S_n n) = a_n k := 
sorry

end general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l2420_242071


namespace ticket_ratio_proof_l2420_242055

-- Define the initial number of tickets Tate has.
def initial_tate_tickets : ℕ := 32

-- Define the additional tickets Tate buys.
def additional_tickets : ℕ := 2

-- Define the total tickets they have together.
def combined_tickets : ℕ := 51

-- Calculate Tate's total number of tickets after buying more tickets.
def total_tate_tickets := initial_tate_tickets + additional_tickets

-- Define the number of tickets Peyton has.
def peyton_tickets := combined_tickets - total_tate_tickets

-- Define the ratio of Peyton's tickets to Tate's tickets.
def tickets_ratio := peyton_tickets / total_tate_tickets

theorem ticket_ratio_proof : tickets_ratio = 1 / 2 :=
by
  unfold tickets_ratio peyton_tickets total_tate_tickets initial_tate_tickets additional_tickets
  norm_num
  sorry

end ticket_ratio_proof_l2420_242055


namespace inequality_solution_l2420_242023

theorem inequality_solution (x : ℝ) : (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 :=
by
  sorry

end inequality_solution_l2420_242023


namespace boys_in_school_l2420_242056

theorem boys_in_school (B G1 G2 : ℕ) (h1 : G1 = 632) (h2 : G2 = G1 + 465) (h3 : G2 = B + 687) : B = 410 :=
by
  sorry

end boys_in_school_l2420_242056


namespace fraction_of_time_l2420_242006

-- Define the time John takes to clean the entire house
def John_time : ℝ := 6

-- Define the combined time it takes Nick and John to clean the entire house
def combined_time : ℝ := 3.6

-- Given this configuration, we need to prove the fraction result.
theorem fraction_of_time (N : ℝ) (H1 : John_time = 6) (H2 : ∀ N, (1/John_time) + (1/N) = 1/combined_time) :
  (John_time / 2) / N = 1 / 3 := 
by sorry

end fraction_of_time_l2420_242006


namespace sum_of_midpoints_x_coordinates_l2420_242095

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l2420_242095


namespace comb_comb_l2420_242004

theorem comb_comb (n1 k1 n2 k2 : ℕ) (h1 : n1 = 10) (h2 : k1 = 3) (h3 : n2 = 8) (h4 : k2 = 4) :
  (Nat.choose n1 k1) * (Nat.choose n2 k2) = 8400 := by
  rw [h1, h2, h3, h4]
  change Nat.choose 10 3 * Nat.choose 8 4 = 8400
  -- Adding the proof steps is not necessary as per instructions
  sorry

end comb_comb_l2420_242004


namespace number_is_209_given_base_value_is_100_l2420_242034

theorem number_is_209_given_base_value_is_100 (n : ℝ) (base_value : ℝ) (H : base_value = 100) (percentage : ℝ) (H1 : percentage = 2.09) : n = 209 :=
by
  sorry

end number_is_209_given_base_value_is_100_l2420_242034


namespace alice_bob_age_difference_18_l2420_242048

-- Define Alice's and Bob's ages with the given constraints
def is_odd (n : ℕ) : Prop := n % 2 = 1

def alice_age (a b : ℕ) : ℕ := 10 * a + b
def bob_age (a b : ℕ) : ℕ := 10 * b + a

theorem alice_bob_age_difference_18 (a b : ℕ) (ha : is_odd a) (hb : is_odd b)
  (h : alice_age a b + 7 = 3 * (bob_age a b + 7)) : alice_age a b - bob_age a b = 18 :=
sorry

end alice_bob_age_difference_18_l2420_242048


namespace exists_n_sum_digits_n3_eq_million_l2420_242015

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem exists_n_sum_digits_n3_eq_million :
  ∃ n : ℕ, sum_digits n = 100 ∧ sum_digits (n ^ 3) = 1000000 := sorry

end exists_n_sum_digits_n3_eq_million_l2420_242015


namespace val_4_at_6_l2420_242079

def at_op (a b : ℤ) : ℤ := 2 * a - 4 * b

theorem val_4_at_6 : at_op 4 6 = -16 := by
  sorry

end val_4_at_6_l2420_242079


namespace max_product_l2420_242078

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l2420_242078


namespace quadratic_function_min_value_l2420_242061

noncomputable def f (a h k : ℝ) (x : ℝ) : ℝ :=
  a * (x - h) ^ 2 + k

theorem quadratic_function_min_value :
  ∀ (f : ℝ → ℝ) (n : ℕ),
  (f n = 13) ∧ (f (n + 1) = 13) ∧ (f (n + 2) = 35) →
  (∃ k, k = 2) :=
  sorry

end quadratic_function_min_value_l2420_242061


namespace bird_families_migration_l2420_242037

theorem bird_families_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (migrated_families : ℕ)
  (remaining_families : ℕ)
  (total_migration_time : ℕ)
  (H1 : total_families = 200)
  (H2 : africa_families = 60)
  (H3 : asia_families = 95)
  (H4 : south_america_families = 30)
  (H5 : africa_days = 7)
  (H6 : asia_days = 14)
  (H7 : south_america_days = 10)
  (H8 : migrated_families = africa_families + asia_families + south_america_families)
  (H9 : remaining_families = total_families - migrated_families)
  (H10 : total_migration_time = 
          africa_families * africa_days + 
          asia_families * asia_days + 
          south_america_families * south_america_days) :
  remaining_families = 15 ∧ total_migration_time = 2050 :=
by
  sorry

end bird_families_migration_l2420_242037


namespace least_four_digit_palindrome_divisible_by_11_l2420_242051

theorem least_four_digit_palindrome_divisible_by_11 : 
  ∃ (A B : ℕ), (A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ 1000 * A + 100 * B + 10 * B + A = 1111 ∧ (2 * A - 2 * B) % 11 = 0) := 
by
  sorry

end least_four_digit_palindrome_divisible_by_11_l2420_242051


namespace scarves_per_box_l2420_242096

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ := 8) 
  (mittens_per_box : ℕ := 6) 
  (total_clothing : ℕ := 80) 
  (total_mittens : ℕ := boxes * mittens_per_box) 
  (total_scarves : ℕ := total_clothing - total_mittens) 
  (scarves_per_box : ℕ := total_scarves / boxes) 
  : scarves_per_box = 4 := 
by 
  sorry

end scarves_per_box_l2420_242096


namespace coeff_exists_l2420_242024

theorem coeff_exists :
  ∃ (A B C : ℕ), 
    ¬(8 ∣ A) ∧ ¬(8 ∣ B) ∧ ¬(8 ∣ C) ∧ 
    (∀ (n : ℕ), 8 ∣ (A * 5^n + B * 3^(n-1) + C))
    :=
sorry

end coeff_exists_l2420_242024


namespace orchid_bushes_planted_l2420_242094

theorem orchid_bushes_planted (b1 b2 : ℕ) (h1 : b1 = 22) (h2 : b2 = 35) : b2 - b1 = 13 :=
by 
  sorry

end orchid_bushes_planted_l2420_242094


namespace graveyard_bones_count_l2420_242003

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end graveyard_bones_count_l2420_242003


namespace probability_diff_color_balls_l2420_242022

theorem probability_diff_color_balls 
  (Box_A_red : ℕ) (Box_A_black : ℕ) (Box_A_white : ℕ) 
  (Box_B_yellow : ℕ) (Box_B_black : ℕ) (Box_B_white : ℕ) 
  (hA : Box_A_red = 3 ∧ Box_A_black = 3 ∧ Box_A_white = 3)
  (hB : Box_B_yellow = 2 ∧ Box_B_black = 2 ∧ Box_B_white = 2) :
  ((Box_A_red * (Box_B_black + Box_B_white + Box_B_yellow))
  + (Box_A_black * (Box_B_yellow + Box_B_white))
  + (Box_A_white * (Box_B_black + Box_B_yellow))) / 
  ((Box_A_red + Box_A_black + Box_A_white) * 
  (Box_B_yellow + Box_B_black + Box_B_white)) = 7 / 9 := 
by
  sorry

end probability_diff_color_balls_l2420_242022


namespace cube_edge_adjacency_l2420_242028

def is_beautiful (f: Finset ℕ) := 
  ∃ a b c d, f = {a, b, c, d} ∧ a = b + c + d

def cube_is_beautiful (faces: Finset (Finset ℕ)) :=
  ∃ t1 t2 t3, t1 ∈ faces ∧ t2 ∈ faces ∧ t3 ∈ faces ∧
  is_beautiful t1 ∧ is_beautiful t2 ∧ is_beautiful t3

def valid_adjacency (v: ℕ) (n1 n2 n3: ℕ) := 
  v = 6 ∧ ((n1 = 2 ∧ n2 = 3 ∧ n3 = 5) ∨
           (n1 = 2 ∧ n2 = 3 ∧ n3 = 7) ∨
           (n1 = 3 ∧ n2 = 5 ∧ n3 = 7))

theorem cube_edge_adjacency : 
  ∀ faces: Finset (Finset ℕ), 
  ∃ v n1 n2 n3, 
  (v = 6 ∧ (valid_adjacency v n1 n2 n3)) ∧
  cube_is_beautiful faces := 
by
  -- Entails the proof, which is not required here
  sorry

end cube_edge_adjacency_l2420_242028


namespace total_number_of_balls_is_twelve_l2420_242033

noncomputable def num_total_balls (a : ℕ) : Prop :=
(3 : ℚ) / a = (25 : ℚ) / 100

theorem total_number_of_balls_is_twelve : num_total_balls 12 :=
by sorry

end total_number_of_balls_is_twelve_l2420_242033


namespace cone_to_cylinder_ratio_l2420_242010

theorem cone_to_cylinder_ratio (r : ℝ) (h_cyl : ℝ) (h_cone : ℝ) 
  (V_cyl : ℝ) (V_cone : ℝ) 
  (h_cyl_eq : h_cyl = 18)
  (r_eq : r = 5)
  (h_cone_eq : h_cone = h_cyl / 3)
  (volume_cyl_eq : V_cyl = π * r^2 * h_cyl)
  (volume_cone_eq : V_cone = 1/3 * π * r^2 * h_cone) :
  V_cone / V_cyl = 1 / 9 := by
  sorry

end cone_to_cylinder_ratio_l2420_242010


namespace bottles_per_case_l2420_242064

theorem bottles_per_case (total_bottles_per_day : ℕ) (cases_required : ℕ) (bottles_per_case : ℕ)
  (h1 : total_bottles_per_day = 65000)
  (h2 : cases_required = 5000) :
  bottles_per_case = total_bottles_per_day / cases_required :=
by
  sorry

end bottles_per_case_l2420_242064


namespace find_number_l2420_242059

theorem find_number (x : ℤ) (h : 4 * x = 28) : x = 7 :=
sorry

end find_number_l2420_242059


namespace one_greater_one_smaller_l2420_242092

theorem one_greater_one_smaller (a b : ℝ) (h : ( (1 + a * b) / (a + b) )^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (b > 1 ∧ -1 < a ∧ a < 1) ∨ (a < -1 ∧ -1 < b ∧ b < 1) ∨ (b < -1 ∧ -1 < a ∧ a < 1) :=
by
  sorry

end one_greater_one_smaller_l2420_242092


namespace abs_a_gt_abs_c_sub_abs_b_l2420_242001

theorem abs_a_gt_abs_c_sub_abs_b (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| :=
sorry

end abs_a_gt_abs_c_sub_abs_b_l2420_242001


namespace total_fish_l2420_242058

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l2420_242058


namespace sum_of_triangles_l2420_242047

def triangle (a b c : ℤ) : ℤ := a + b - c

theorem sum_of_triangles : triangle 1 3 4 + triangle 2 5 6 = 1 := by
  sorry

end sum_of_triangles_l2420_242047


namespace percentage_increase_l2420_242016

variable (A B y : ℝ)

theorem percentage_increase (h1 : B > A) (h2 : A > 0) :
  B = A + y / 100 * A ↔ y = 100 * (B - A) / A :=
by
  sorry

end percentage_increase_l2420_242016


namespace pizza_eating_group_l2420_242087

theorem pizza_eating_group (x y : ℕ) (h1 : 6 * x + 2 * y ≥ 49) (h2 : 7 * x + 3 * y ≤ 59) : x = 8 ∧ y = 2 := by
  sorry

end pizza_eating_group_l2420_242087


namespace determine_A_l2420_242085

noncomputable def is_single_digit (n : ℕ) : Prop := n < 10

theorem determine_A (A B C : ℕ) (hABC : 3 * (100 * A + 10 * B + C) = 888)
  (hA_single_digit : is_single_digit A) (hB_single_digit : is_single_digit B) (hC_single_digit : is_single_digit C)
  (h_different : A ≠ B ∧ B ≠ C ∧ A ≠ C) : A = 2 := 
  sorry

end determine_A_l2420_242085


namespace find_a8_l2420_242049

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def geom_sequence (a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_geom_sequence (S a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)

def arithmetic_sequence (S : ℕ → ℝ) :=
  S 9 = S 3 + S 6

def sum_a2_a5 (a : ℕ → ℝ) :=
  a 2 + a 5 = 4

theorem find_a8 (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)
  (hgeom_seq : geom_sequence a a1 q)
  (hsum_geom_seq : sum_geom_sequence S a a1 q)
  (harith_seq : arithmetic_sequence S)
  (hsum_a2_a5 : sum_a2_a5 a) :
  a 8 = 2 :=
sorry

end find_a8_l2420_242049


namespace tiles_per_row_l2420_242039

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end tiles_per_row_l2420_242039


namespace p_n_div_5_iff_not_mod_4_zero_l2420_242018

theorem p_n_div_5_iff_not_mod_4_zero (n : ℕ) (h : 0 < n) : 
  (1 + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
by {
  sorry
}

end p_n_div_5_iff_not_mod_4_zero_l2420_242018


namespace tencent_technological_innovation_basis_tencent_innovative_development_analysis_l2420_242066

-- Define the dialectical materialist basis conditions
variable (dialectical_negation essence_innovation development_perspective unity_of_opposites : Prop)

-- Define Tencent's emphasis on technological innovation
variable (tencent_innovation : Prop)

-- Define the relationship between Tencent's development and materialist view of development
variable (unity_of_things_developmental progressiveness_tortuosity quantitative_qualitative_changes : Prop)
variable (tencent_development : Prop)

-- Prove that Tencent's emphasis on technological innovation aligns with dialectical materialism
theorem tencent_technological_innovation_basis :
  dialectical_negation ∧ essence_innovation ∧ development_perspective ∧ unity_of_opposites → tencent_innovation :=
by sorry

-- Prove that Tencent's innovative development aligns with dialectical materialist view of development
theorem tencent_innovative_development_analysis :
  unity_of_things_developmental ∧ progressiveness_tortuosity ∧ quantitative_qualitative_changes → tencent_development :=
by sorry

end tencent_technological_innovation_basis_tencent_innovative_development_analysis_l2420_242066


namespace transformed_solution_equiv_l2420_242060

noncomputable def quadratic_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 0}

noncomputable def transformed_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (10^x) > 0}

theorem transformed_solution_equiv (f : ℝ → ℝ) :
  quadratic_solution_set f = {x | x < -1 ∨ x > 1 / 2} →
  transformed_solution_set f = {x | x > -Real.log 2} :=
by sorry

end transformed_solution_equiv_l2420_242060


namespace longer_side_length_l2420_242065

theorem longer_side_length (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x * y = 221) : max x y = 17 :=
by
  sorry

end longer_side_length_l2420_242065


namespace Ms_Rush_Speed_to_be_on_time_l2420_242021

noncomputable def required_speed (d t r : ℝ) :=
  d = 50 * (t + 1/12) ∧ 
  d = 70 * (t - 1/9) →
  r = d / t →
  r = 74

theorem Ms_Rush_Speed_to_be_on_time 
  (d t r : ℝ) 
  (h1 : d = 50 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/9)) 
  (h3 : r = d / t) : 
  r = 74 :=
sorry

end Ms_Rush_Speed_to_be_on_time_l2420_242021


namespace carpet_rate_proof_l2420_242091

noncomputable def carpet_rate (breadth_first : ℝ) (length_ratio : ℝ) (cost_second : ℝ) : ℝ :=
  let length_first := length_ratio * breadth_first
  let area_first := length_first * breadth_first
  let length_second := length_first * 1.4
  let breadth_second := breadth_first * 1.25
  let area_second := length_second * breadth_second 
  cost_second / area_second

theorem carpet_rate_proof : carpet_rate 6 1.44 4082.4 = 45 :=
by
  -- Here we provide the goal and state what needs to be proven.
  sorry

end carpet_rate_proof_l2420_242091


namespace A_n_is_integer_l2420_242082

open Real

noncomputable def A_n (a b : ℕ) (θ : ℝ) (n : ℕ) : ℝ :=
  (a^2 + b^2)^n * sin (n * θ)

theorem A_n_is_integer (a b : ℕ) (h : a > b) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < pi/2) (h_sin : sin θ = 2 * a * b / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, A_n a b θ n = k :=
by
  sorry

end A_n_is_integer_l2420_242082


namespace AM_GM_HM_inequality_l2420_242098

theorem AM_GM_HM_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (2 * a * b) / (a + b) := 
sorry

end AM_GM_HM_inequality_l2420_242098


namespace tom_distance_before_karen_wins_l2420_242029

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end tom_distance_before_karen_wins_l2420_242029


namespace gain_percent_calculation_l2420_242014

theorem gain_percent_calculation (gain_paise : ℕ) (cost_price_rupees : ℕ) (rupees_to_paise : ℕ)
  (h_gain_paise : gain_paise = 70)
  (h_cost_price_rupees : cost_price_rupees = 70)
  (h_rupees_to_paise : rupees_to_paise = 100) :
  ((gain_paise / rupees_to_paise) / cost_price_rupees) * 100 = 1 :=
by
  -- Placeholder to indicate the need for proof
  sorry

end gain_percent_calculation_l2420_242014


namespace unique_polynomial_P_l2420_242052

open Polynomial

/-- The only polynomial P with real coefficients such that
    xP(y/x) + yP(x/y) = x + y for all nonzero real numbers x and y 
    is P(x) = x. --/
theorem unique_polynomial_P (P : ℝ[X]) (hP : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x * P.eval (y / x) + y * P.eval (x / y) = x + y) :
P = Polynomial.C 1 * X :=
by sorry

end unique_polynomial_P_l2420_242052


namespace evaluate_expression_l2420_242050

theorem evaluate_expression : 12 * ((1/3 : ℚ) + (1/4) + (1/6))⁻¹ = 16 := 
by 
  sorry

end evaluate_expression_l2420_242050


namespace johns_total_spending_l2420_242046

theorem johns_total_spending
    (online_phone_price : ℝ := 2000)
    (phone_price_increase : ℝ := 0.02)
    (phone_case_price : ℝ := 35)
    (screen_protector_price : ℝ := 15)
    (accessories_discount : ℝ := 0.05)
    (sales_tax : ℝ := 0.06) :
    let store_phone_price := online_phone_price * (1 + phone_price_increase)
    let regular_accessories_price := phone_case_price + screen_protector_price
    let discounted_accessories_price := regular_accessories_price * (1 - accessories_discount)
    let pre_tax_total := store_phone_price + discounted_accessories_price
    let total_spending := pre_tax_total * (1 + sales_tax)
    total_spending = 2212.75 :=
by
    sorry

end johns_total_spending_l2420_242046


namespace range_of_a_l2420_242009

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a (a : ℝ) :
  (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a p - f a q) / (p - q) > 1)
  ↔ 3 ≤ a :=
by
  sorry

end range_of_a_l2420_242009


namespace nancy_coffee_expense_l2420_242012

-- Definitions corresponding to the conditions
def cost_double_espresso : ℝ := 3.00
def cost_iced_coffee : ℝ := 2.50
def days : ℕ := 20

-- The statement of the problem
theorem nancy_coffee_expense :
  (days * (cost_double_espresso + cost_iced_coffee)) = 110.00 := by
  sorry

end nancy_coffee_expense_l2420_242012


namespace problem_statement_l2420_242099

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end problem_statement_l2420_242099


namespace n_divisible_by_40_l2420_242072

theorem n_divisible_by_40 {n : ℕ} (h_pos : 0 < n)
  (h1 : ∃ k1 : ℕ, 2 * n + 1 = k1 * k1)
  (h2 : ∃ k2 : ℕ, 3 * n + 1 = k2 * k2) :
  ∃ k : ℕ, n = 40 * k := 
sorry

end n_divisible_by_40_l2420_242072


namespace prime_odd_sum_l2420_242041

theorem prime_odd_sum (x y : ℕ) (h_prime : Prime x) (h_odd : y % 2 = 1) (h_eq : x^2 + y = 2005) : x + y = 2003 :=
by
  sorry

end prime_odd_sum_l2420_242041


namespace n_not_composite_l2420_242013

theorem n_not_composite
  (n : ℕ) (h1 : n > 1)
  (a : ℕ) (q : ℕ) (hq_prime : Nat.Prime q)
  (hq1 : q ∣ (n - 1))
  (hq2 : q > Nat.sqrt n - 1)
  (hn_div : n ∣ (a^(n-1) - 1))
  (hgcd : Nat.gcd (a^(n-1)/q - 1) n = 1) :
  ¬ Nat.Prime n :=
sorry

end n_not_composite_l2420_242013


namespace metal_waste_l2420_242019

theorem metal_waste (a b : ℝ) (h : a < b) :
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * radius^2
  let side_square := a / Real.sqrt 2
  let area_square := side_square^2
  area_rectangle - area_square = a * b - ( a ^ 2 ) / 2 := by
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * (radius ^ 2)
  let side_square := a / Real.sqrt 2
  let area_square := side_square ^ 2
  sorry

end metal_waste_l2420_242019


namespace work_completion_time_l2420_242063

theorem work_completion_time (x : ℝ) (a_work_rate b_work_rate combined_work_rate : ℝ) :
  a_work_rate = 1 / 15 ∧
  b_work_rate = 1 / 20 ∧
  combined_work_rate = 1 / 7.2 ∧
  a_work_rate + b_work_rate + (1 / x) = combined_work_rate → 
  x = 45 := by
  sorry

end work_completion_time_l2420_242063


namespace smallest_sum_of_squares_l2420_242040

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l2420_242040
