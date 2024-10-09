import Mathlib

namespace vectors_parallel_l466_46676

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end vectors_parallel_l466_46676


namespace sum_of_quotient_and_reciprocal_l466_46614

theorem sum_of_quotient_and_reciprocal (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 500) : 
    (x / y + y / x) = 41 / 20 := 
sorry

end sum_of_quotient_and_reciprocal_l466_46614


namespace distinct_students_count_l466_46625

-- Definition of the initial parameters
def num_gauss : Nat := 12
def num_euler : Nat := 10
def num_fibonnaci : Nat := 7
def overlap : Nat := 1

-- The main theorem to prove
theorem distinct_students_count : num_gauss + num_euler + num_fibonnaci - overlap = 28 := by
  sorry

end distinct_students_count_l466_46625


namespace total_price_of_books_l466_46678

theorem total_price_of_books (total_books: ℕ) (math_books: ℕ) (math_book_cost: ℕ) (history_book_cost: ℕ) (price: ℕ) 
  (h1 : total_books = 90) 
  (h2 : math_books = 54) 
  (h3 : math_book_cost = 4) 
  (h4 : history_book_cost = 5)
  (h5 : price = 396) :
  let history_books := total_books - math_books
  let math_books_price := math_books * math_book_cost
  let history_books_price := history_books * history_book_cost
  let total_price := math_books_price + history_books_price
  total_price = price := 
  by
    sorry

end total_price_of_books_l466_46678


namespace total_students_in_Lansing_l466_46659

def n_schools : Nat := 25
def students_per_school : Nat := 247
def total_students : Nat := n_schools * students_per_school

theorem total_students_in_Lansing :
  total_students = 6175 :=
  by
    -- we can either compute manually or just put sorry for automated assistance
    sorry

end total_students_in_Lansing_l466_46659


namespace lisa_goal_l466_46607

theorem lisa_goal 
  (total_quizzes : ℕ) 
  (target_percentage : ℝ) 
  (completed_quizzes : ℕ) 
  (earned_A : ℕ) 
  (remaining_quizzes : ℕ) : 
  total_quizzes = 40 → 
  target_percentage = 0.9 → 
  completed_quizzes = 25 → 
  earned_A = 20 → 
  remaining_quizzes = (total_quizzes - completed_quizzes) → 
  (earned_A + remaining_quizzes ≥ target_percentage * total_quizzes) → 
  remaining_quizzes - (total_quizzes * target_percentage - earned_A) = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lisa_goal_l466_46607


namespace mrs_heine_dogs_l466_46680

theorem mrs_heine_dogs (total_biscuits biscuits_per_dog : ℕ) (h1 : total_biscuits = 6) (h2 : biscuits_per_dog = 3) :
  total_biscuits / biscuits_per_dog = 2 :=
by
  sorry

end mrs_heine_dogs_l466_46680


namespace ratio_final_to_original_l466_46660

-- Given conditions
variable (d : ℝ)
variable (h1 : 364 = d * 1.30)

-- Problem statement
theorem ratio_final_to_original : (364 / d) = 1.3 := 
by sorry

end ratio_final_to_original_l466_46660


namespace couple_slices_each_l466_46638

noncomputable def slices_for_couple (total_slices children_slices people_in_couple : ℕ) : ℕ :=
  (total_slices - children_slices) / people_in_couple

theorem couple_slices_each (people_in_couple children slices_per_pizza num_pizzas : ℕ) (H1 : people_in_couple = 2) (H2 : children = 6) (H3 : slices_per_pizza = 4) (H4 : num_pizzas = 3) :
  slices_for_couple (num_pizzas * slices_per_pizza) (children * 1) people_in_couple = 3 := 
  by
  rw [H1, H2, H3, H4]
  show slices_for_couple (3 * 4) (6 * 1) 2 = 3
  rfl

end couple_slices_each_l466_46638


namespace find_distance_l466_46662

-- Definitions based on conditions
def speed : ℝ := 75 -- in km/hr
def time : ℝ := 4 -- in hr

-- Statement to be proved
theorem find_distance : speed * time = 300 := by
  sorry

end find_distance_l466_46662


namespace diminished_value_is_seven_l466_46650

theorem diminished_value_is_seven (x y : ℕ) (hx : x = 280)
  (h_eq : x / 5 + 7 = x / 4 - y) : y = 7 :=
by {
  sorry
}

end diminished_value_is_seven_l466_46650


namespace solve_equation_l466_46682

theorem solve_equation (a b : ℚ) : 
  ((b = 0) → false) ∧ 
  ((4 * a - 3 = 0) → ((5 * b - 1 = 0) → a = 3 / 4 ∧ b = 1 / 5)) ∧ 
  ((4 * a - 3 ≠ 0) → (∃ x : ℚ, x = (5 * b - 1) / (4 * a - 3))) :=
by
  sorry

end solve_equation_l466_46682


namespace solved_fraction_equation_l466_46655

theorem solved_fraction_equation :
  ∀ (x : ℚ),
    x ≠ 2 →
    x ≠ 7 →
    x ≠ -5 →
    (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) →
    x = 55 / 13 := by
  sorry

end solved_fraction_equation_l466_46655


namespace bonus_implies_completion_l466_46664

variable (John : Type)
variable (completes_all_tasks_perfectly : John → Prop)
variable (receives_bonus : John → Prop)

theorem bonus_implies_completion :
  (∀ e : John, completes_all_tasks_perfectly e → receives_bonus e) →
  (∀ e : John, receives_bonus e → completes_all_tasks_perfectly e) :=
by
  intros h e
  sorry

end bonus_implies_completion_l466_46664


namespace find_values_l466_46696

theorem find_values (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 = 4 * a * b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end find_values_l466_46696


namespace largest_possible_cylindrical_tank_radius_in_crate_l466_46641

theorem largest_possible_cylindrical_tank_radius_in_crate
  (crate_length : ℝ) (crate_width : ℝ) (crate_height : ℝ)
  (cylinder_height : ℝ) (cylinder_radius : ℝ)
  (h_cube : crate_length = 20 ∧ crate_width = 20 ∧ crate_height = 20)
  (h_cylinder_in_cube : cylinder_height = 20 ∧ 2 * cylinder_radius ≤ 20) :
  cylinder_radius = 10 :=
sorry

end largest_possible_cylindrical_tank_radius_in_crate_l466_46641


namespace cost_of_saddle_l466_46621

theorem cost_of_saddle (S : ℝ) (H : 4 * S + S = 5000) : S = 1000 :=
by sorry

end cost_of_saddle_l466_46621


namespace simplify_and_evaluate_expr_l466_46656

theorem simplify_and_evaluate_expr 
  (x : ℝ) 
  (h : x = 1/2) : 
  (2 * x - 1) ^ 2 - (3 * x + 1) * (3 * x - 1) + 5 * x * (x - 1) = -5 / 2 := 
by
  sorry

end simplify_and_evaluate_expr_l466_46656


namespace polynomial_condition_l466_46694

noncomputable def polynomial_of_degree_le (n : ℕ) (P : Polynomial ℝ) :=
  P.degree ≤ n

noncomputable def has_nonneg_coeff (P : Polynomial ℝ) :=
  ∀ i, 0 ≤ P.coeff i

theorem polynomial_condition
  (n : ℕ) (P : Polynomial ℝ)
  (h1 : polynomial_of_degree_le n P)
  (h2 : has_nonneg_coeff P)
  (h3 : ∀ x : ℝ, x > 0 → P.eval x * P.eval (1 / x) ≤ (P.eval 1) ^ 2) : 
  ∃ a_n : ℝ, 0 ≤ a_n ∧ P = Polynomial.C a_n * Polynomial.X^n :=
sorry

end polynomial_condition_l466_46694


namespace four_transformations_of_1989_l466_46684

-- Definition of the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Initial number
def initial_number : ℕ := 1989

-- Theorem statement
theorem four_transformations_of_1989 : 
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits initial_number))) = 9 :=
by
  sorry

end four_transformations_of_1989_l466_46684


namespace students_neither_class_l466_46677

theorem students_neither_class : 
  let total_students := 1500
  let music_students := 300
  let art_students := 200
  let dance_students := 100
  let theater_students := 50
  let music_art_students := 80
  let music_dance_students := 40
  let music_theater_students := 30
  let art_dance_students := 25
  let art_theater_students := 20
  let dance_theater_students := 10
  let music_art_dance_students := 50
  let music_art_theater_students := 30
  let art_dance_theater_students := 20
  let music_dance_theater_students := 10
  let all_four_students := 5
  total_students - 
    (music_students + 
     art_students + 
     dance_students + 
     theater_students - 
     (music_art_students + 
      music_dance_students + 
      music_theater_students + 
      art_dance_students + 
      art_theater_students + 
      dance_theater_students) + 
     (music_art_dance_students + 
      music_art_theater_students + 
      art_dance_theater_students + 
      music_dance_theater_students) - 
     all_four_students) = 950 :=
sorry

end students_neither_class_l466_46677


namespace find_number_l466_46629

theorem find_number 
  (a b c d : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 6 * a + 9 * b + 3 * c + d = 88)
  (h6 : a - b + c - d = -6)
  (h7 : a - 9 * b + 3 * c - d = -46) : 
  1000 * a + 100 * b + 10 * c + d = 6507 := 
sorry

end find_number_l466_46629


namespace find_m_value_l466_46673

theorem find_m_value (m : ℤ) (h : (∀ x : ℤ, (x-5)*(x+7) = x^2 - mx - 35)) : m = -2 :=
by sorry

end find_m_value_l466_46673


namespace factorize_expression_l466_46681

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l466_46681


namespace ellipse_focus_distance_l466_46637

theorem ellipse_focus_distance : ∀ (x y : ℝ), 9 * x^2 + y^2 = 900 → 2 * Real.sqrt (10^2 - 30^2) = 40 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end ellipse_focus_distance_l466_46637


namespace sum_of_fractions_limit_one_l466_46606

theorem sum_of_fractions_limit_one :
  (∑' (a : ℕ), ∑' (b : ℕ), (1 : ℝ) / ((a + 1) : ℝ) ^ (b + 1)) = 1 := 
sorry

end sum_of_fractions_limit_one_l466_46606


namespace determine_m_ratio_l466_46646

def ratio_of_C_to_A_investment (x : ℕ) (m : ℕ) (total_gain : ℕ) (a_share : ℕ) : Prop :=
  total_gain = 18000 ∧ a_share = 6000 ∧
  (12 * x / (12 * x + 4 * m * x) = 1 / 3)

theorem determine_m_ratio (x : ℕ) (m : ℕ) (h : ratio_of_C_to_A_investment x m 18000 6000) :
  m = 6 :=
by
  sorry

end determine_m_ratio_l466_46646


namespace find_certain_number_l466_46691

theorem find_certain_number (x : ℝ) 
  (h : 3889 + x - 47.95000000000027 = 3854.002) : x = 12.95200000000054 :=
by
  sorry

end find_certain_number_l466_46691


namespace fish_caught_in_second_catch_l466_46668

theorem fish_caught_in_second_catch {N x : ℕ} (hN : N = 1750) (hx1 : 70 * x = 2 * N) : x = 50 :=
by
  sorry

end fish_caught_in_second_catch_l466_46668


namespace conditional_two_exits_one_effective_l466_46613

def conditional_structure (decide : Bool) : Prop :=
  if decide then True else False

theorem conditional_two_exits_one_effective (decide : Bool) :
  conditional_structure decide ↔ True :=
by
  sorry

end conditional_two_exits_one_effective_l466_46613


namespace infinite_equal_pairs_of_equal_terms_l466_46602

theorem infinite_equal_pairs_of_equal_terms {a : ℤ → ℤ}
  (h : ∀ n, a n = (a (n - 1) + a (n + 1)) / 4)
  (i j : ℤ) (hij : a i = a j) :
  ∃ (infinitely_many_pairs : ℕ → ℤ × ℤ), ∀ k, a (infinitely_many_pairs k).1 = a (infinitely_many_pairs k).2 :=
sorry

end infinite_equal_pairs_of_equal_terms_l466_46602


namespace wrapping_paper_area_correct_l466_46633

noncomputable def wrapping_paper_area (l w h : ℝ) (hlw : l ≥ w) : ℝ :=
  (l + 2*h)^2

theorem wrapping_paper_area_correct (l w h : ℝ) (hlw : l ≥ w) :
  wrapping_paper_area l w h hlw = (l + 2*h)^2 :=
by
  sorry

end wrapping_paper_area_correct_l466_46633


namespace simplify_polynomial_l466_46630

variable (x : ℝ)

theorem simplify_polynomial :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6)
  = 5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 := by
  sorry

end simplify_polynomial_l466_46630


namespace jacket_total_selling_price_l466_46699

theorem jacket_total_selling_price :
  let original_price := 120
  let discount_rate := 0.30
  let tax_rate := 0.08
  let processing_fee := 5
  let discounted_price := original_price * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  let total_price := discounted_price + tax + processing_fee
  total_price = 95.72 := by
  sorry

end jacket_total_selling_price_l466_46699


namespace average_goods_per_hour_l466_46666

-- Define the conditions
def morning_goods : ℕ := 64
def morning_hours : ℕ := 4
def afternoon_rate : ℕ := 23
def afternoon_hours : ℕ := 3

-- Define the target statement to be proven
theorem average_goods_per_hour : (morning_goods + afternoon_rate * afternoon_hours) / (morning_hours + afternoon_hours) = 19 := by
  -- Add proof steps here
  sorry

end average_goods_per_hour_l466_46666


namespace neg_p_sufficient_but_not_necessary_for_q_l466_46692

variable {x : ℝ}

def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

theorem neg_p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, ¬ p x → q x) ∧ ¬ (∀ x : ℝ, q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_but_not_necessary_for_q_l466_46692


namespace total_pairs_of_shoes_equivalence_l466_46690

variable (Scott Anthony Jim Melissa Tim: ℕ)

theorem total_pairs_of_shoes_equivalence
    (h1 : Scott = 7)
    (h2 : Anthony = 3 * Scott)
    (h3 : Jim = Anthony - 2)
    (h4 : Jim = 2 * Melissa)
    (h5 : Tim = (Anthony + Melissa) / 2):

  Scott + Anthony + Jim + Melissa + Tim = 71 :=
  by
  sorry

end total_pairs_of_shoes_equivalence_l466_46690


namespace number_of_ways_to_feed_animals_l466_46688

-- Definitions for the conditions
def pairs_of_animals := 5
def alternating_feeding (start_with_female : Bool) (remaining_pairs : ℕ) : ℕ :=
if start_with_female then
  (pairs_of_animals.factorial / 2 ^ pairs_of_animals)
else
  0 -- we can ignore this case as it is not needed

-- Theorem statement
theorem number_of_ways_to_feed_animals :
  alternating_feeding true pairs_of_animals = 2880 :=
sorry

end number_of_ways_to_feed_animals_l466_46688


namespace find_value_of_t_l466_46657

variable (a b v d t r : ℕ)

-- All variables are non-zero digits (1-9)
axiom non_zero_a : 0 < a ∧ a < 10
axiom non_zero_b : 0 < b ∧ b < 10
axiom non_zero_v : 0 < v ∧ v < 10
axiom non_zero_d : 0 < d ∧ d < 10
axiom non_zero_t : 0 < t ∧ t < 10
axiom non_zero_r : 0 < r ∧ r < 10

-- Given conditions
axiom condition1 : a + b = v
axiom condition2 : v + d = t
axiom condition3 : t + a = r
axiom condition4 : b + d + r = 18

theorem find_value_of_t : t = 9 :=
by sorry

end find_value_of_t_l466_46657


namespace panthers_score_l466_46647

-- Definitions as per the conditions
def total_points (C P : ℕ) : Prop := C + P = 48
def margin (C P : ℕ) : Prop := C = P + 20

-- Theorem statement proving Panthers score 14 points
theorem panthers_score (C P : ℕ) (h1 : total_points C P) (h2 : margin C P) : P = 14 :=
sorry

end panthers_score_l466_46647


namespace jake_peaches_calculation_l466_46648

variable (S_p : ℕ) (J_p : ℕ)

-- Given that Steven has 19 peaches
def steven_peaches : ℕ := 19

-- Jake has 12 fewer peaches than Steven
def jake_peaches : ℕ := S_p - 12

theorem jake_peaches_calculation (h1 : S_p = steven_peaches) (h2 : S_p = 19) :
  J_p = jake_peaches := 
by
  sorry

end jake_peaches_calculation_l466_46648


namespace soccer_team_players_l466_46670

theorem soccer_team_players
  (first_half_starters : ℕ)
  (first_half_subs : ℕ)
  (second_half_mult : ℕ)
  (did_not_play : ℕ)
  (players_prepared : ℕ) :
  first_half_starters = 11 →
  first_half_subs = 2 →
  second_half_mult = 2 →
  did_not_play = 7 →
  players_prepared = 20 :=
by
  -- Proof steps go here
  sorry

end soccer_team_players_l466_46670


namespace gain_percent_is_30_l466_46698

-- Given conditions
def CostPrice : ℕ := 100
def SellingPrice : ℕ := 130
def Gain : ℕ := SellingPrice - CostPrice
def GainPercent : ℕ := (Gain * 100) / CostPrice

-- The theorem to be proven
theorem gain_percent_is_30 :
  GainPercent = 30 := sorry

end gain_percent_is_30_l466_46698


namespace tangent_line_slope_through_origin_l466_46603

theorem tangent_line_slope_through_origin :
  (∃ a : ℝ, (a^3 + a + 16 = (3 * a^2 + 1) * a ∧ a = 2)) →
  (3 * (2 : ℝ)^2 + 1 = 13) :=
by
  intro h
  -- Detailed proof goes here
  sorry

end tangent_line_slope_through_origin_l466_46603


namespace sum_valid_two_digit_integers_l466_46675

theorem sum_valid_two_digit_integers :
  ∃ S : ℕ, S = 36 ∧ (∀ n, 10 ≤ n ∧ n < 100 →
    (∃ a b, n = 10 * a + b ∧ a + b ∣ n ∧ 2 * a * b ∣ n → n = 36)) :=
by
  sorry

end sum_valid_two_digit_integers_l466_46675


namespace geometric_sequence_an_l466_46631

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 3 else 3 * (2:ℝ)^(n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  if n = 1 then 3 else (3 * (2:ℝ)^n - 3)

theorem geometric_sequence_an (n : ℕ) (h1 : a 1 = 3) (h2 : S 2 = 9) :
  a n = 3 * 2^(n-1) ∧ S n = 3 * (2^n - 1) :=
by
  sorry

end geometric_sequence_an_l466_46631


namespace problem_statement_l466_46689

theorem problem_statement : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end problem_statement_l466_46689


namespace outer_boundary_diameter_l466_46636

def width_jogging_path : ℝ := 4
def width_garden_ring : ℝ := 10
def diameter_pond : ℝ := 12

theorem outer_boundary_diameter : 2 * (diameter_pond / 2 + width_garden_ring + width_jogging_path) = 40 := by
  sorry

end outer_boundary_diameter_l466_46636


namespace min_val_of_f_l466_46663

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end min_val_of_f_l466_46663


namespace triangle_circle_area_relation_l466_46642

theorem triangle_circle_area_relation (A B C : ℝ) (h : 15^2 + 20^2 = 25^2) (A_area_eq : A + B + 150 = C) :
  A + B + 150 = C :=
by
  -- The proof has been omitted.
  sorry

end triangle_circle_area_relation_l466_46642


namespace rectangle_perimeter_l466_46626

-- Definitions based on conditions
def length (w : ℝ) : ℝ := 2 * w
def width (w : ℝ) : ℝ := w
def area (w : ℝ) : ℝ := length w * width w
def perimeter (w : ℝ) : ℝ := 2 * (length w + width w)

-- Problem statement: Prove that the perimeter is 120 cm given area is 800 cm² and length is twice the width
theorem rectangle_perimeter (w : ℝ) (h : area w = 800) : perimeter w = 120 := by
  sorry

end rectangle_perimeter_l466_46626


namespace monotonic_f_on_interval_l466_46651

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 10) - 2

theorem monotonic_f_on_interval : 
  ∀ x y : ℝ, 
    x ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    y ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    x ≤ y → 
    f x ≤ f y :=
sorry

end monotonic_f_on_interval_l466_46651


namespace Paul_dig_days_alone_l466_46653

/-- Jake's daily work rate -/
def Jake_work_rate : ℚ := 1 / 16

/-- Hari's daily work rate -/
def Hari_work_rate : ℚ := 1 / 48

/-- Combined work rate of Jake, Paul, and Hari, when they work together they can dig the well in 8 days -/
def combined_work_rate (Paul_work_rate : ℚ) : Prop :=
  Jake_work_rate + Paul_work_rate + Hari_work_rate = 1 / 8

/-- Theorem stating that Paul can dig the well alone in 24 days -/
theorem Paul_dig_days_alone : ∃ (P : ℚ), combined_work_rate (1 / P) ∧ P = 24 :=
by
  use 24
  unfold combined_work_rate
  sorry

end Paul_dig_days_alone_l466_46653


namespace james_coursework_materials_expense_l466_46622

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end james_coursework_materials_expense_l466_46622


namespace gcd_pow_minus_one_l466_46620

theorem gcd_pow_minus_one (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  Nat.gcd (2^m - 1) (2^n - 1) = 2^(Nat.gcd m n) - 1 :=
by
  sorry

end gcd_pow_minus_one_l466_46620


namespace total_percent_decrease_l466_46652

theorem total_percent_decrease (initial_value : ℝ) (val1 val2 : ℝ) :
  initial_value > 0 →
  val1 = initial_value * (1 - 0.60) →
  val2 = val1 * (1 - 0.10) →
  (initial_value - val2) / initial_value * 100 = 64 :=
by
  intros h_initial h_val1 h_val2
  sorry

end total_percent_decrease_l466_46652


namespace calculation_result_l466_46609

theorem calculation_result:
  (-1:ℤ)^3 - 8 / (-2) + 4 * abs (-5) = 23 := by
  sorry

end calculation_result_l466_46609


namespace jellyfish_cost_l466_46644

theorem jellyfish_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : J = 20 := by
  sorry

end jellyfish_cost_l466_46644


namespace compute_expression_l466_46683

theorem compute_expression : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end compute_expression_l466_46683


namespace great_circle_bisects_angle_l466_46661

noncomputable def north_pole : Point := sorry
noncomputable def equator_point (C : Point) : Prop := sorry
noncomputable def great_circle_through (P Q : Point) : Circle := sorry
noncomputable def equidistant_from_N (A B N : Point) : Prop := sorry
noncomputable def spherical_triangle (A B C : Point) : Triangle := sorry
noncomputable def bisects_angle (C N A B : Point) : Prop := sorry

theorem great_circle_bisects_angle
  (N A B C: Point)
  (hN: N = north_pole)
  (hA: equidistant_from_N A B N)
  (hC: equator_point C)
  (hTriangle: spherical_triangle A B C)
  : bisects_angle C N A B :=
sorry

end great_circle_bisects_angle_l466_46661


namespace men_earnings_l466_46601

-- Definitions based on given problem conditions
variables (M rm W rw B rb X : ℝ)
variables (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) -- positive quantities
variables (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180)

-- The theorem we want to prove
theorem men_earnings (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180) : 
  ∃ men_earnings : ℝ, men_earnings = 5 * M * rm :=
by 
  -- Proof is omitted
  exact Exists.intro (5 * M * rm) rfl

end men_earnings_l466_46601


namespace computer_price_decrease_l466_46616

theorem computer_price_decrease 
  (initial_price : ℕ) 
  (decrease_factor : ℚ)
  (years : ℕ) 
  (final_price : ℕ) 
  (h1 : initial_price = 8100)
  (h2 : decrease_factor = 1/3)
  (h3 : years = 6)
  (h4 : final_price = 2400) : 
  initial_price * (1 - decrease_factor) ^ (years / 2) = final_price :=
by
  sorry

end computer_price_decrease_l466_46616


namespace segment_length_of_points_A_l466_46640

-- Define the basic setup
variable (d BA CA : ℝ)
variable {A B C : Point} -- Assume we have a type Point for the geometric points

-- Establish some conditions: A right triangle with given lengths
def is_right_triangle (A B C : Point) : Prop := sorry -- Placeholder for definition

def distance (P Q : Point) : ℝ := sorry -- Placeholder for the distance function

-- Conditions
variables (h_right_triangle : is_right_triangle A B C)
variables (h_hypotenuse : distance B C = d)
variables (h_smallest_leg : min (distance B A) (distance C A) = min BA CA)

-- The theorem statement
theorem segment_length_of_points_A (h_right_triangle : is_right_triangle A B C)
                                    (h_hypotenuse : distance B C = d)
                                    (h_smallest_leg : min (distance B A) (distance C A) = min BA CA) :
  ∃ A, (∀ t : ℝ, distance O A = d - min BA CA) :=
sorry -- Proof to be provided

end segment_length_of_points_A_l466_46640


namespace solve_for_xy_l466_46643

theorem solve_for_xy (x y : ℕ) : 
  (4^x / 2^(x + y) = 16) ∧ (9^(x + y) / 3^(5 * y) = 81) → x * y = 32 :=
by
  sorry

end solve_for_xy_l466_46643


namespace find_tangent_line_at_neg1_l466_46608

noncomputable def tangent_line (x : ℝ) : ℝ := 2 * x^2 + 3

theorem find_tangent_line_at_neg1 :
  let x := -1
  let m := 4 * x
  let y := 2 * x^2 + 3
  let tangent := y + m * (x - x)
  tangent = -4 * x + 1 :=
by
  sorry

end find_tangent_line_at_neg1_l466_46608


namespace sum_of_squares_l466_46679

theorem sum_of_squares (a d : Int) : 
  ∃ y1 y2 : Int, a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (3*a + y1*d)^2 + (a + y2*d)^2 :=
by
  sorry

end sum_of_squares_l466_46679


namespace triangle_side_ratio_l466_46634

theorem triangle_side_ratio (a b c : ℝ) (h1 : a + b ≤ 2 * c) (h2 : b + c ≤ 3 * a) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  2 / 3 < c / a ∧ c / a < 2 :=
by
  sorry

end triangle_side_ratio_l466_46634


namespace meryll_questions_l466_46632

/--
Meryll wants to write a total of 35 multiple-choice questions and 15 problem-solving questions. 
She has written \(\frac{2}{5}\) of the multiple-choice questions and \(\frac{1}{3}\) of the problem-solving questions.
We need to prove that she needs to write 31 more questions in total.
-/
theorem meryll_questions : (35 - (2 / 5) * 35) + (15 - (1 / 3) * 15) = 31 := by
  sorry

end meryll_questions_l466_46632


namespace trajectory_equation_l466_46665

-- Define the fixed points F1 and F2
structure Point where
  x : ℝ
  y : ℝ

def F1 : Point := ⟨-2, 0⟩
def F2 : Point := ⟨2, 0⟩

-- Define the moving point M and the condition it must satisfy
def satisfies_condition (M : Point) : Prop :=
  (Real.sqrt ((M.x + 2)^2 + M.y^2) - Real.sqrt ((M.x - 2)^2 + M.y^2)) = 4

-- The trajectory of the point M must satisfy y = 0 and x >= 2
def on_trajectory (M : Point) : Prop :=
  M.y = 0 ∧ M.x ≥ 2

-- The final theorem to be proved
theorem trajectory_equation (M : Point) (h : satisfies_condition M) : on_trajectory M := by
  sorry

end trajectory_equation_l466_46665


namespace sqrt_sum_eq_l466_46628

theorem sqrt_sum_eq :
  (Real.sqrt (9 / 2) + Real.sqrt (2 / 9)) = (11 * Real.sqrt 2 / 6) :=
sorry

end sqrt_sum_eq_l466_46628


namespace number_of_ways_to_form_divisible_number_l466_46687

def valid_digits : List ℕ := [0, 2, 4, 7, 8, 9]

def is_divisible_by_4 (d1 d2 : ℕ) : Prop :=
  (d1 * 10 + d2) % 4 = 0

def is_divisible_by_3 (sum_of_digits : ℕ) : Prop :=
  sum_of_digits % 3 = 0

def replace_asterisks_to_form_divisible_number : Prop :=
  ∃ (a1 a2 a3 a4 a5 l : ℕ), a1 ∈ valid_digits ∧ a2 ∈ valid_digits ∧ a3 ∈ valid_digits ∧ a4 ∈ valid_digits ∧ a5 ∈ valid_digits ∧
  l ∈ [0, 2, 4, 8] ∧
  is_divisible_by_4 0 l ∧
  is_divisible_by_3 (11 + a1 + a2 + a3 + a4 + a5) ∧
  (4 * 324 = 1296)

theorem number_of_ways_to_form_divisible_number :
  replace_asterisks_to_form_divisible_number :=
  sorry

end number_of_ways_to_form_divisible_number_l466_46687


namespace distinctPaintedCubeConfigCount_l466_46627

-- Define a painted cube with given face colors
structure PaintedCube where
  blue_face : ℤ
  yellow_faces : Finset ℤ
  red_faces : Finset ℤ
  -- Ensure logical conditions about faces
  face_count : blue_face ∉ yellow_faces ∧ blue_face ∉ red_faces ∧
               yellow_faces ∩ red_faces = ∅ ∧ yellow_faces.card = 2 ∧
               red_faces.card = 3

-- There are no orientation-invariant rotations that change the configuration
def equivPaintedCube (c1 c2 : PaintedCube) : Prop :=
  ∃ (r: ℤ), 
    -- rotate c1 by r to get c2
    true -- placeholder for rotation logic

-- The set of all possible distinct painted cubes under rotation constraints is defined
def possibleConfigurations : Finset PaintedCube :=
  sorry  -- construct this set considering rotations

-- The main proposition
theorem distinctPaintedCubeConfigCount : (possibleConfigurations.card = 4) :=
  sorry

end distinctPaintedCubeConfigCount_l466_46627


namespace lemon_heads_per_package_l466_46658

theorem lemon_heads_per_package (total_lemon_heads boxes : ℕ)
  (H : total_lemon_heads = 54)
  (B : boxes = 9)
  (no_leftover : total_lemon_heads % boxes = 0) :
  total_lemon_heads / boxes = 6 :=
sorry

end lemon_heads_per_package_l466_46658


namespace intersection_M_N_l466_46604

theorem intersection_M_N :
  let M := { x : ℝ | abs x ≤ 2 }
  let N := {-1, 0, 2, 3}
  M ∩ N = {-1, 0, 2} :=
by
  sorry

end intersection_M_N_l466_46604


namespace principal_amount_l466_46695

/-- Given:
 - 820 = P + (P * R * 2) / 100
 - 1020 = P + (P * R * 6) / 100
Prove:
 - P = 720
--/

theorem principal_amount (P R : ℝ) (h1 : 820 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 6) / 100) : P = 720 :=
by
  sorry

end principal_amount_l466_46695


namespace lottery_win_probability_l466_46619

theorem lottery_win_probability :
  let MegaBall_prob := 1 / 30
  let WinnerBall_prob := 1 / Nat.choose 50 5
  let BonusBall_prob := 1 / 15
  let Total_prob := MegaBall_prob * WinnerBall_prob * BonusBall_prob
  Total_prob = 1 / 953658000 :=
by
  sorry

end lottery_win_probability_l466_46619


namespace additional_discount_during_sale_l466_46697

theorem additional_discount_during_sale:
  ∀ (list_price : ℝ) (max_typical_discount_pct : ℝ) (lowest_possible_sale_pct : ℝ),
  30 ≤ max_typical_discount_pct ∧ max_typical_discount_pct ≤ 50 ∧
  lowest_possible_sale_pct = 40 ∧ 
  list_price = 80 →
  ((max_typical_discount_pct * list_price / 100) - (lowest_possible_sale_pct * list_price / 100)) * 100 / 
    (max_typical_discount_pct * list_price / 100) = 20 :=
by
  sorry

end additional_discount_during_sale_l466_46697


namespace initial_bottles_count_l466_46615

theorem initial_bottles_count : 
  ∀ (jason_buys harry_buys bottles_left initial_bottles : ℕ), 
  jason_buys = 5 → 
  harry_buys = 6 → 
  bottles_left = 24 → 
  initial_bottles = bottles_left + jason_buys + harry_buys → 
  initial_bottles = 35 :=
by
  intros jason_buys harry_buys bottles_left initial_bottles
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end initial_bottles_count_l466_46615


namespace exceed_1000_cents_l466_46674

def total_amount (n : ℕ) : ℕ :=
  3 * (3 ^ n - 1) / (3 - 1)

theorem exceed_1000_cents : 
  ∃ n : ℕ, total_amount n ≥ 1000 ∧ (n + 7) % 7 = 6 := 
by
  sorry

end exceed_1000_cents_l466_46674


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l466_46611

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l466_46611


namespace g_at_2_l466_46667

def g (x : ℝ) : ℝ := x^3 - x

theorem g_at_2 : g 2 = 6 :=
by
  sorry

end g_at_2_l466_46667


namespace original_weight_of_apples_l466_46623

theorem original_weight_of_apples (x : ℕ) (h1 : 5 * (x - 30) = 2 * x) : x = 50 :=
by
  sorry

end original_weight_of_apples_l466_46623


namespace t_of_polynomial_has_factor_l466_46686

theorem t_of_polynomial_has_factor (t : ℤ) :
  (∃ a b : ℤ, x ^ 3 - x ^ 2 - 7 * x + t = (x + 1) * (x ^ 2 + a * x + b)) → t = -5 :=
by
  sorry

end t_of_polynomial_has_factor_l466_46686


namespace puzzle_piece_total_l466_46685

theorem puzzle_piece_total :
  let p1 := 1000
  let p2 := p1 + 0.30 * p1
  let p3 := 2 * p2
  let p4 := (p1 + p3) + 0.50 * (p1 + p3)
  let p5 := 3 * p4
  let p6 := p1 + p2 + p3 + p4 + p5
  p1 + p2 + p3 + p4 + p5 + p6 = 55000
:= sorry

end puzzle_piece_total_l466_46685


namespace remainder_when_sum_div_by_3_l466_46671

theorem remainder_when_sum_div_by_3 
  (m n p q : ℕ)
  (a : ℕ := 6 * m + 4)
  (b : ℕ := 6 * n + 4)
  (c : ℕ := 6 * p + 4)
  (d : ℕ := 6 * q + 4)
  : (a + b + c + d) % 3 = 1 :=
by
  sorry

end remainder_when_sum_div_by_3_l466_46671


namespace marbles_lost_correct_l466_46645

-- Define the initial number of marbles
def initial_marbles : ℕ := 16

-- Define the current number of marbles
def current_marbles : ℕ := 9

-- Define the number of marbles lost
def marbles_lost (initial current : ℕ) : ℕ := initial - current

-- State the proof problem: Given the conditions, prove the number of marbles lost is 7
theorem marbles_lost_correct : marbles_lost initial_marbles current_marbles = 7 := by
  sorry

end marbles_lost_correct_l466_46645


namespace range_of_a_if_inequality_holds_l466_46693

noncomputable def satisfies_inequality_for_all_xy_pos (a : ℝ) :=
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9

theorem range_of_a_if_inequality_holds :
  (∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9) → (a ≥ 4) :=
by
  sorry

end range_of_a_if_inequality_holds_l466_46693


namespace complete_the_square_l466_46612

theorem complete_the_square (y : ℝ) : (y^2 + 12*y + 40) = (y + 6)^2 + 4 := by
  sorry

end complete_the_square_l466_46612


namespace annual_interest_approx_l466_46669

noncomputable def P : ℝ := 10000
noncomputable def r : ℝ := 0.05
noncomputable def t : ℝ := 1
noncomputable def e : ℝ := Real.exp 1

theorem annual_interest_approx :
  let A := P * Real.exp (r * t)
  let interest := A - P
  abs (interest - 512.71) < 0.01 := sorry

end annual_interest_approx_l466_46669


namespace phase_and_initial_phase_theorem_l466_46605

open Real

noncomputable def phase_and_initial_phase (x : ℝ) : ℝ := 3 * sin (-x + π / 6)

theorem phase_and_initial_phase_theorem :
  ∃ φ : ℝ, ∃ ψ : ℝ,
    ∀ x : ℝ, phase_and_initial_phase x = 3 * sin (x + φ) ∧
    (φ = 5 * π / 6) ∧ (ψ = φ) :=
sorry

end phase_and_initial_phase_theorem_l466_46605


namespace triangle_perimeters_sum_l466_46624

theorem triangle_perimeters_sum :
  ∃ (t : ℕ),
    (∀ (A B C D : Type) (x y : ℕ), 
      (AB = 7 ∧ BC = 17 ∧ AD = x ∧ CD = x ∧ BD = y ∧ x^2 - y^2 = 240) →
      t = 114) :=
sorry

end triangle_perimeters_sum_l466_46624


namespace steve_travel_time_l466_46654

theorem steve_travel_time :
  ∀ (d : ℕ) (v_back : ℕ) (v_to : ℕ),
  d = 20 →
  v_back = 10 →
  v_to = v_back / 2 →
  d / v_to + d / v_back = 6 := 
by
  intros d v_back v_to h1 h2 h3
  sorry

end steve_travel_time_l466_46654


namespace green_apples_count_l466_46639

-- Definitions for the conditions in the problem
def total_apples : ℕ := 19
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

-- Statement expressing that the number of green apples on the table is 2
theorem green_apples_count : (total_apples - red_apples - yellow_apples = 2) :=
by
  sorry

end green_apples_count_l466_46639


namespace calc1_calc2_calc3_calc4_l466_46617

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end calc1_calc2_calc3_calc4_l466_46617


namespace abs_of_neg_one_third_l466_46672

theorem abs_of_neg_one_third : abs (- (1 / 3)) = (1 / 3) := by
  sorry

end abs_of_neg_one_third_l466_46672


namespace find_m_l466_46649

open Real

noncomputable def vec_a : ℝ × ℝ := (-1, 2)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (m, 3)

theorem find_m (m : ℝ) (h : -1 * m + 2 * 3 = 0) : m = 6 :=
sorry

end find_m_l466_46649


namespace find_k_l466_46610

def f (a b c x : Int) : Int := a * x^2 + b * x + c

theorem find_k (a b c k : Int)
  (h₁ : f a b c 2 = 0)
  (h₂ : 100 < f a b c 7 ∧ f a b c 7 < 110)
  (h₃ : 120 < f a b c 8 ∧ f a b c 8 < 130)
  (h₄ : 6000 * k < f a b c 100 ∧ f a b c 100 < 6000 * (k + 1)) :
  k = 0 := 
sorry

end find_k_l466_46610


namespace circles_intersect_l466_46600

def circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 4 * p.2 + 3 = 0}

theorem circles_intersect : ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 :=
by
  sorry

end circles_intersect_l466_46600


namespace gasoline_price_increase_l466_46635

theorem gasoline_price_increase 
  (highest_price : ℝ) (lowest_price : ℝ) 
  (h_high : highest_price = 17) 
  (h_low : lowest_price = 10) : 
  (highest_price - lowest_price) / lowest_price * 100 = 70 := 
by
  /- proof can go here -/
  sorry

end gasoline_price_increase_l466_46635


namespace beautiful_39th_moment_l466_46618

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end beautiful_39th_moment_l466_46618
