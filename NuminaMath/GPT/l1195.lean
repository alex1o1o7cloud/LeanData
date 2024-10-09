import Mathlib

namespace find_n_l1195_119547

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) = 9) : n = 2 :=
by
  sorry

end find_n_l1195_119547


namespace total_apples_l1195_119579

variable (A : ℕ)
variables (too_small not_ripe perfect : ℕ)

-- Conditions
axiom small_fraction : too_small = A / 6
axiom ripe_fraction  : not_ripe = A / 3
axiom remaining_fraction : perfect = A / 2
axiom perfect_count : perfect = 15

theorem total_apples : A = 30 :=
sorry

end total_apples_l1195_119579


namespace symmetric_origin_a_minus_b_l1195_119545

noncomputable def A (a : ℝ) := (a, -2)
noncomputable def B (b : ℝ) := (4, b)
def symmetric (p q : ℝ × ℝ) : Prop := (q.1 = -p.1) ∧ (q.2 = -p.2)

theorem symmetric_origin_a_minus_b (a b : ℝ) (hA : A a = (-4, -2)) (hB : B b = (4, 2)) :
  a - b = -6 := by
  sorry

end symmetric_origin_a_minus_b_l1195_119545


namespace measure_six_liters_l1195_119543

-- Given conditions as constants
def container_capacity : ℕ := 40
def ten_liter_bucket_capacity : ℕ := 10
def nine_liter_jug_capacity : ℕ := 9
def five_liter_jug_capacity : ℕ := 5

-- Goal: Measure out exactly 6 liters of milk using the above containers
theorem measure_six_liters (container : ℕ) (ten_bucket : ℕ) (nine_jug : ℕ) (five_jug : ℕ) :
  container = 40 →
  ten_bucket ≤ 10 →
  nine_jug ≤ 9 →
  five_jug ≤ 5 →
  ∃ (sequence_of_steps : ℕ → ℕ) (final_ten_bucket : ℕ),
    final_ten_bucket = 6 ∧ final_ten_bucket ≤ ten_bucket :=
by
  intro hcontainer hten_bucket hnine_jug hfive_jug
  sorry

end measure_six_liters_l1195_119543


namespace smallest_whole_number_larger_than_sum_l1195_119555

noncomputable def mixed_number1 : ℚ := 3 + 2/3
noncomputable def mixed_number2 : ℚ := 4 + 1/4
noncomputable def mixed_number3 : ℚ := 5 + 1/5
noncomputable def mixed_number4 : ℚ := 6 + 1/6
noncomputable def mixed_number5 : ℚ := 7 + 1/7

noncomputable def sum_of_mixed_numbers : ℚ :=
  mixed_number1 + mixed_number2 + mixed_number3 + mixed_number4 + mixed_number5

theorem smallest_whole_number_larger_than_sum : 
  ∃ n : ℤ, (n : ℚ) > sum_of_mixed_numbers ∧ n = 27 :=
by
  sorry

end smallest_whole_number_larger_than_sum_l1195_119555


namespace total_notes_l1195_119593

theorem total_notes (total_amount : ℤ) (num_50_notes : ℤ) (value_50 : ℤ) (value_500 : ℤ) (total_notes : ℤ) :
  total_amount = num_50_notes * value_50 + (total_notes - num_50_notes) * value_500 → 
  total_amount = 10350 → num_50_notes = 77 → value_50 = 50 → value_500 = 500 → total_notes = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_notes_l1195_119593


namespace circle_equation_l1195_119505

theorem circle_equation :
  ∃ x y : ℝ, x = 2 ∧ y = 0 ∧ ∀ (p q : ℝ), ((p - x)^2 + q^2 = 4) ↔ (p^2 + q^2 - 4 * p = 0) :=
sorry

end circle_equation_l1195_119505


namespace evaluate_expression_l1195_119522

theorem evaluate_expression : (6^6) * (12^6) * (6^12) * (12^12) = 72^18 := 
by sorry

end evaluate_expression_l1195_119522


namespace cos_A_value_find_c_l1195_119527

theorem cos_A_value (a b c A B C : ℝ) (h : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) : 
  Real.cos A = 1 / 2 := 
  sorry

theorem find_c (B C : ℝ) (A : B + C = Real.pi - A) (h1 : 1 = 1) 
  (h2 : Real.cos (B / 2) * Real.cos (B / 2) + Real.cos (C / 2) * Real.cos (C / 2) = 1 + Real.sqrt (3) / 4) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt (3) / 3 ∨ c = Real.sqrt (3) / 3 := 
  sorry

end cos_A_value_find_c_l1195_119527


namespace marie_lost_erasers_l1195_119567

def initialErasers : ℕ := 95
def finalErasers : ℕ := 53

theorem marie_lost_erasers : initialErasers - finalErasers = 42 := by
  sorry

end marie_lost_erasers_l1195_119567


namespace total_history_and_maths_l1195_119589

-- Defining the conditions
def total_students : ℕ := 25
def fraction_like_maths : ℚ := 2 / 5
def fraction_like_science : ℚ := 1 / 3

-- Theorem statement
theorem total_history_and_maths : (total_students * fraction_like_maths + (total_students * (1 - fraction_like_maths) * (1 - fraction_like_science))) = 20 := by
  sorry

end total_history_and_maths_l1195_119589


namespace correct_statements_l1195_119506

-- Definitions based on the conditions and question
def S (n : ℕ) : ℤ := -n^2 + 7 * n + 1

-- Definition of the sequence an
def a (n : ℕ) : ℤ := 
  if n = 1 then 7 
  else S n - S (n - 1)

-- Theorem statements based on the correct answers derived from solution
theorem correct_statements :
  (∀ n : ℕ, n > 4 → a n < 0) ∧ (S 3 = S 4 ∧ (∀ m : ℕ, S m ≤ S 3)) :=
by {
  sorry
}

end correct_statements_l1195_119506


namespace general_term_formula_l1195_119566

theorem general_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 3^n - 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n - 1)) →
  a 1 = 2 →
  ∀ n, a n = 2 * 3^(n - 1) :=
by
    intros hS ha h1 n
    sorry

end general_term_formula_l1195_119566


namespace gcf_120_180_240_is_60_l1195_119514

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end gcf_120_180_240_is_60_l1195_119514


namespace train_length_is_330_meters_l1195_119598

noncomputable def train_speed : ℝ := 60 -- in km/hr
noncomputable def man_speed : ℝ := 6    -- in km/hr
noncomputable def time : ℝ := 17.998560115190788  -- in seconds

noncomputable def relative_speed_km_per_hr : ℝ := train_speed + man_speed
noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def relative_speed_m_per_s : ℝ := 
  relative_speed_km_per_hr * conversion_factor

theorem train_length_is_330_meters : 
  (relative_speed_m_per_s * time) = 330 := 
sorry

end train_length_is_330_meters_l1195_119598


namespace expression_D_is_odd_l1195_119560

namespace ProofProblem

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem expression_D_is_odd :
  is_odd (3 + 5 + 1) :=
by
  sorry

end ProofProblem

end expression_D_is_odd_l1195_119560


namespace positive_solutions_l1195_119594

theorem positive_solutions (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔
  x = 1 ∨ x = 3 :=
by
  sorry

end positive_solutions_l1195_119594


namespace total_expenditure_l1195_119586

-- Define the conditions
def cost_per_acre : ℕ := 20
def acres_bought : ℕ := 30
def house_cost : ℕ := 120000
def cost_per_cow : ℕ := 1000
def cows_bought : ℕ := 20
def cost_per_chicken : ℕ := 5
def chickens_bought : ℕ := 100
def hourly_installation_cost : ℕ := 100
def installation_hours : ℕ := 6
def solar_equipment_cost : ℕ := 6000

-- Define the total cost breakdown
def land_cost : ℕ := cost_per_acre * acres_bought
def cows_cost : ℕ := cost_per_cow * cows_bought
def chickens_cost : ℕ := cost_per_chicken * chickens_bought
def solar_installation_cost : ℕ := (hourly_installation_cost * installation_hours) + solar_equipment_cost

-- Define the total cost
def total_cost : ℕ :=
  land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost

-- The theorem statement
theorem total_expenditure : total_cost = 147700 :=
by
  -- Proof steps would go here
  sorry

end total_expenditure_l1195_119586


namespace games_new_friends_l1195_119562

-- Definitions based on the conditions
def total_games_all_friends : ℕ := 141
def games_old_friends : ℕ := 53

-- Statement of the problem
theorem games_new_friends {games_new_friends : ℕ} :
  games_new_friends = total_games_all_friends - games_old_friends :=
sorry

end games_new_friends_l1195_119562


namespace polynomial_equality_l1195_119515

theorem polynomial_equality (x y : ℝ) (h₁ : 3 * x + 2 * y = 6) (h₂ : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := 
by
  sorry

end polynomial_equality_l1195_119515


namespace amoeba_after_ten_days_l1195_119565

def amoeba_count (n : ℕ) : ℕ := 
  3^n

theorem amoeba_after_ten_days : amoeba_count 10 = 59049 := 
by
  -- proof omitted
  sorry

end amoeba_after_ten_days_l1195_119565


namespace cubic_conversion_l1195_119536

theorem cubic_conversion (h : 1 = 100) : 1 = 1000000 :=
by
  sorry

end cubic_conversion_l1195_119536


namespace contradiction_even_odd_l1195_119564

theorem contradiction_even_odd (a b c : ℕ) :
  (∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (¬((x % 2 = 0 ∧ y % 2 ≠ 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 = 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 ≠ 0 ∧ z % 2 = 0)))) → false :=
by
  sorry

end contradiction_even_odd_l1195_119564


namespace ranking_Fiona_Giselle_Ella_l1195_119544

-- Definitions of scores 
variable (score : String → ℕ)

-- Conditions based on the problem statement
def ella_not_highest : Prop := ¬ (score "Ella" = max (score "Ella") (max (score "Fiona") (score "Giselle")))
def giselle_not_lowest : Prop := ¬ (score "Giselle" = min (score "Ella") (score "Giselle"))

-- The goal is to rank the scores from highest to lowest
def score_ranking : Prop := (score "Fiona" > score "Giselle") ∧ (score "Giselle" > score "Ella")

theorem ranking_Fiona_Giselle_Ella :
  ella_not_highest score →
  giselle_not_lowest score →
  score_ranking score :=
by
  sorry

end ranking_Fiona_Giselle_Ella_l1195_119544


namespace remaining_distance_l1195_119528

theorem remaining_distance (speed time distance_covered total_distance remaining_distance : ℕ) 
  (h1 : speed = 60) 
  (h2 : time = 2) 
  (h3 : total_distance = 300)
  (h4 : distance_covered = speed * time) 
  (h5 : remaining_distance = total_distance - distance_covered) : 
  remaining_distance = 180 := 
by
  sorry

end remaining_distance_l1195_119528


namespace largest_expr_l1195_119577

noncomputable def A : ℝ := 2 * 1005 ^ 1006
noncomputable def B : ℝ := 1005 ^ 1006
noncomputable def C : ℝ := 1004 * 1005 ^ 1005
noncomputable def D : ℝ := 2 * 1005 ^ 1005
noncomputable def E : ℝ := 1005 ^ 1005
noncomputable def F : ℝ := 1005 ^ 1004

theorem largest_expr : A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by
  sorry

end largest_expr_l1195_119577


namespace bug_traverses_36_tiles_l1195_119534

-- Define the dimensions of the rectangle and the bug's problem setup
def width : ℕ := 12
def length : ℕ := 25

-- Define the function to calculate the number of tiles traversed by the bug
def tiles_traversed (w l : ℕ) : ℕ :=
  w + l - Nat.gcd w l

-- Prove the number of tiles traversed by the bug is 36
theorem bug_traverses_36_tiles : tiles_traversed width length = 36 :=
by
  -- This part will be proven; currently, we add sorry
  sorry

end bug_traverses_36_tiles_l1195_119534


namespace exterior_angle_regular_polygon_l1195_119592

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l1195_119592


namespace largest_integer_square_two_digits_l1195_119529

theorem largest_integer_square_two_digits : 
  ∃ M : ℤ, (M * M ≥ 10 ∧ M * M < 100) ∧ (∀ x : ℤ, (x * x ≥ 10 ∧ x * x < 100) → x ≤ M) ∧ M = 9 := 
by
  sorry

end largest_integer_square_two_digits_l1195_119529


namespace time_to_run_above_tree_l1195_119591

-- Defining the given conditions
def tiger_length : ℕ := 5
def tree_trunk_length : ℕ := 20
def time_to_pass_grass : ℕ := 1

-- Defining the speed of the tiger
def tiger_speed : ℕ := tiger_length / time_to_pass_grass

-- Defining the total distance the tiger needs to run
def total_distance : ℕ := tree_trunk_length + tiger_length

-- The theorem stating the time it takes for the tiger to run above the fallen tree trunk
theorem time_to_run_above_tree :
  (total_distance / tiger_speed) = 5 :=
by
  -- Trying to fit the solution steps as formal Lean statements
  sorry

end time_to_run_above_tree_l1195_119591


namespace coloring_impossible_l1195_119559

-- Define vertices for the outer pentagon and inner star
inductive Vertex
| A | B | C | D | E | A' | B' | C' | D' | E'

open Vertex

-- Define segments in the figure
def Segments : List (Vertex × Vertex) :=
  [(A, B), (B, C), (C, D), (D, E), (E, A),
   (A, A'), (B, B'), (C, C'), (D, D'), (E, E'),
   (A', C), (C, E'), (E, B'), (B, D'), (D, A')]

-- Color type
inductive Color
| Red | Green | Blue

open Color

-- Condition for coloring: no two segments of the same color share a common endpoint
def distinct_color (c : Vertex → Color) : Prop :=
  ∀ (v1 v2 v3 : Vertex) (h1 : (v1, v2) ∈ Segments) (h2 : (v2, v3) ∈ Segments),
  c v1 ≠ c v2 ∧ c v2 ≠ c v3 ∧ c v1 ≠ c v3

-- Statement of the proof problem
theorem coloring_impossible : ¬ ∃ (c : Vertex → Color), distinct_color c := 
by 
  sorry

end coloring_impossible_l1195_119559


namespace rate_of_current_l1195_119518

/-- The speed of a boat in still water is 20 km/hr, and the rate of current is c km/hr.
    The distance travelled downstream in 24 minutes is 9.2 km. What is the rate of the current? -/
theorem rate_of_current (c : ℝ) (h : 24/60 = 0.4 ∧ 9.2 = (20 + c) * 0.4) : c = 3 :=
by
  sorry  -- Proof is not required, only the statement is necessary.

end rate_of_current_l1195_119518


namespace roots_greater_than_two_range_l1195_119561

theorem roots_greater_than_two_range (m : ℝ) :
  ∀ x1 x2 : ℝ, (x1^2 + (m - 4) * x1 + 6 - m = 0) ∧ (x2^2 + (m - 4) * x2 + 6 - m = 0) ∧ (x1 > 2) ∧ (x2 > 2) →
  -2 < m ∧ m ≤ 2 - 2 * Real.sqrt 3 :=
by
  sorry

end roots_greater_than_two_range_l1195_119561


namespace brown_ball_weight_l1195_119500

def total_weight : ℝ := 9.12
def weight_blue : ℝ := 6
def weight_brown : ℝ := 3.12

theorem brown_ball_weight : total_weight - weight_blue = weight_brown :=
by 
  sorry

end brown_ball_weight_l1195_119500


namespace correct_calculation_l1195_119541

-- Define the statements for each option
def option_A (a : ℕ) : Prop := (a^2)^3 = a^5
def option_B (a : ℕ) : Prop := a^3 + a^2 = a^6
def option_C (a : ℕ) : Prop := a^6 / a^3 = a^3
def option_D (a : ℕ) : Prop := a^3 * a^2 = a^6

-- Define the theorem stating that option C is the only correct one
theorem correct_calculation (a : ℕ) : ¬option_A a ∧ ¬option_B a ∧ option_C a ∧ ¬option_D a := by
  sorry

end correct_calculation_l1195_119541


namespace perfect_square_of_expression_l1195_119595

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem perfect_square_of_expression : 
  (∃ k : ℕ, (factorial 19 * 2 = k ∧ (factorial 20 * factorial 19) / 5 = k * k)) := sorry

end perfect_square_of_expression_l1195_119595


namespace area_difference_l1195_119504

-- Define the areas of individual components
def area_of_square : ℕ := 1
def area_of_small_triangle : ℚ := (1 / 2) * area_of_square
def area_of_large_triangle : ℚ := (1 / 2) * (1 * 2 * area_of_square)

-- Define the total area of the first figure
def first_figure_area : ℚ := 
    8 * area_of_square +
    6 * area_of_small_triangle +
    2 * area_of_large_triangle

-- Define the total area of the second figure
def second_figure_area : ℚ := 
    4 * area_of_square +
    6 * area_of_small_triangle +
    8 * area_of_large_triangle

-- Define the statement to prove the difference in areas
theorem area_difference : second_figure_area - first_figure_area = 2 := by
    -- sorry is used to indicate that the proof is omitted
    sorry

end area_difference_l1195_119504


namespace least_xy_value_l1195_119508

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end least_xy_value_l1195_119508


namespace soccer_lineup_count_l1195_119571

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem soccer_lineup_count : 
  let total_players := 18
  let goalies := 1
  let defenders := 6
  let forwards := 4
  18 * choose 17 6 * choose 11 4 = 73457760 :=
by
  sorry

end soccer_lineup_count_l1195_119571


namespace magic_square_l1195_119501

variable (a b c d e s: ℕ)

axiom h1 : 30 + e + 18 = s
axiom h2 : 15 + c + d = s
axiom h3 : a + 27 + b = s
axiom h4 : 30 + 15 + a = s
axiom h5 : e + c + 27 = s
axiom h6 : 18 + d + b = s
axiom h7 : 30 + c + b = s
axiom h8 : a + c + 18 = s

theorem magic_square : d + e = 47 :=
by
  sorry

end magic_square_l1195_119501


namespace jack_final_apples_l1195_119578

-- Jack's transactions and initial count as conditions
def initial_count : ℕ := 150
def sold_to_jill : ℕ := initial_count * 30 / 100
def remaining_after_jill : ℕ := initial_count - sold_to_jill
def sold_to_june : ℕ := remaining_after_jill * 20 / 100
def remaining_after_june : ℕ := remaining_after_jill - sold_to_june
def donated_to_charity : ℕ := 5
def final_count : ℕ := remaining_after_june - donated_to_charity

-- Proof statement
theorem jack_final_apples : final_count = 79 := by
  sorry

end jack_final_apples_l1195_119578


namespace total_biscuits_needed_l1195_119580

-- Definitions
def number_of_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem statement
theorem total_biscuits_needed : number_of_dogs * biscuits_per_dog = 6 :=
by sorry

end total_biscuits_needed_l1195_119580


namespace structure_cube_count_l1195_119532

theorem structure_cube_count :
  let middle_layer := 16
  let other_layers := 4 * 24
  middle_layer + other_layers = 112 :=
by
  let middle_layer := 16
  let other_layers := 4 * 24
  have h : middle_layer + other_layers = 112 := by
    sorry
  exact h

end structure_cube_count_l1195_119532


namespace range_of_k_tan_alpha_l1195_119570

noncomputable def f (x k : Real) : Real := Real.sin x + k

theorem range_of_k (k : Real) : 
  (∃ x : Real, f x k = 1) ↔ (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem tan_alpha (α k : Real) (h : α ∈ Set.Ioo (0 : Real) Real.pi) (hf : f α k = 1 / 3 + k) : 
  Real.tan α = Real.sqrt 2 / 4 :=
sorry

end range_of_k_tan_alpha_l1195_119570


namespace geom_seq_decreasing_l1195_119582

theorem geom_seq_decreasing :
  (∀ n : ℕ, (4 : ℝ) * 3^(1 - (n + 1) : ℤ) < (4 : ℝ) * 3^(1 - n : ℤ)) :=
sorry

end geom_seq_decreasing_l1195_119582


namespace factorial_division_l1195_119549

theorem factorial_division (n : ℕ) (h : n = 9) : n.factorial / (n - 1).factorial = 9 :=
by 
  rw [h]
  sorry

end factorial_division_l1195_119549


namespace ratio_of_small_rectangle_length_to_width_l1195_119510

-- Define the problem conditions
variables (s : ℝ)

-- Define the length and width of the small rectangle
def length_of_small_rectangle := 3 * s
def width_of_small_rectangle := s

-- Prove that the ratio of the length to the width of the small rectangle is 3
theorem ratio_of_small_rectangle_length_to_width : 
  length_of_small_rectangle s / width_of_small_rectangle s = 3 :=
by
  sorry

end ratio_of_small_rectangle_length_to_width_l1195_119510


namespace norma_initial_cards_l1195_119569

def initial_card_count (lost: ℕ) (left: ℕ) : ℕ :=
  lost + left

theorem norma_initial_cards : initial_card_count 70 18 = 88 :=
  by
    -- skipping proof
    sorry

end norma_initial_cards_l1195_119569


namespace conditional_probability_l1195_119558

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end conditional_probability_l1195_119558


namespace num_rectangular_tables_l1195_119563

theorem num_rectangular_tables (R : ℕ) 
  (rectangular_tables_seat : R * 10 = 70) :
  R = 7 := by
  sorry

end num_rectangular_tables_l1195_119563


namespace sum_of_digits_of_N_l1195_119525

theorem sum_of_digits_of_N :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧ 
    5879 % N = 14 ∧ 
    ((N / 10) + (N % 10)) = 8 := 
sorry

end sum_of_digits_of_N_l1195_119525


namespace smallest_four_digit_remainder_l1195_119583

theorem smallest_four_digit_remainder :
  ∃ N : ℕ, (N % 6 = 5) ∧ (1000 ≤ N ∧ N ≤ 9999) ∧ (∀ M : ℕ, (M % 6 = 5) ∧ (1000 ≤ M ∧ M ≤ 9999) → N ≤ M) ∧ N = 1001 :=
by
  sorry

end smallest_four_digit_remainder_l1195_119583


namespace product_of_numbers_l1195_119575

theorem product_of_numbers :
  ∃ (a b c : ℚ), a + b + c = 30 ∧
                 a = 2 * (b + c) ∧
                 b = 5 * c ∧
                 a + c = 22 ∧
                 a * b * c = 2500 / 9 :=
by
  sorry

end product_of_numbers_l1195_119575


namespace min_value_a_l1195_119597

theorem min_value_a (a : ℕ) :
  (6 * (a + 1)) / (a^2 + 8 * a + 6) ≤ 1 / 100 ↔ a ≥ 594 := sorry

end min_value_a_l1195_119597


namespace clock_minutes_to_correct_time_l1195_119531

def slow_clock_time_ratio : ℚ := 14 / 15

noncomputable def slow_clock_to_correct_time (slow_clock_time : ℚ) : ℚ :=
  slow_clock_time / slow_clock_time_ratio

theorem clock_minutes_to_correct_time :
  slow_clock_to_correct_time 14 = 15 :=
by
  sorry

end clock_minutes_to_correct_time_l1195_119531


namespace least_positive_integer_special_property_l1195_119552

/-- 
  Prove that 9990 is the least positive integer whose digits sum to a multiple of 27 
  and the number itself is not a multiple of 27.
-/
theorem least_positive_integer_special_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (Nat.digits 10 n).sum % 27 = 0 ∧ 
  n % 27 ≠ 0 ∧ 
  ∀ m : ℕ, (m > 0 ∧ (Nat.digits 10 m).sum % 27 = 0 ∧ m % 27 ≠ 0 → n ≤ m) := 
by
  sorry

end least_positive_integer_special_property_l1195_119552


namespace pastrami_sandwich_cost_l1195_119557

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l1195_119557


namespace find_y_minus_x_l1195_119530

theorem find_y_minus_x (x y : ℝ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : y - x = 7.5 :=
by
  sorry

end find_y_minus_x_l1195_119530


namespace line_slope_intercept_l1195_119513

theorem line_slope_intercept (x y : ℝ) (k b : ℝ) (h : 3 * x + 4 * y + 5 = 0) :
  k = -3 / 4 ∧ b = -5 / 4 :=
by sorry

end line_slope_intercept_l1195_119513


namespace total_students_in_class_l1195_119584

theorem total_students_in_class (F G B N T : ℕ)
  (hF : F = 41)
  (hG : G = 22)
  (hB : B = 9)
  (hN : N = 15)
  (hT : T = (F + G - B) + N) :
  T = 69 :=
by
  -- This is a theorem statement, proof is intentionally omitted.
  sorry

end total_students_in_class_l1195_119584


namespace sum_of_three_numbers_l1195_119553

theorem sum_of_three_numbers (x : ℝ) (a b c : ℝ) (h1 : a = 5 * x) (h2 : b = x) (h3 : c = 4 * x) (h4 : c = 400) :
  a + b + c = 1000 := by
  sorry

end sum_of_three_numbers_l1195_119553


namespace roses_in_vase_l1195_119576

theorem roses_in_vase (initial_roses added_roses : ℕ) (h₀ : initial_roses = 10) (h₁ : added_roses = 8) : initial_roses + added_roses = 18 :=
by
  sorry

end roses_in_vase_l1195_119576


namespace max_regions_by_five_lines_l1195_119502

theorem max_regions_by_five_lines : 
  ∀ (R : ℕ → ℕ), R 1 = 2 → R 2 = 4 → (∀ n, R (n + 1) = R n + (n + 1)) → R 5 = 16 :=
by
  intros R hR1 hR2 hRec
  sorry

end max_regions_by_five_lines_l1195_119502


namespace jake_weight_l1195_119507

theorem jake_weight:
  ∃ (J S : ℝ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196) :=
by
  sorry

end jake_weight_l1195_119507


namespace expression_evaluation_l1195_119520

noncomputable def evaluate_expression (a b c : ℚ) : ℚ :=
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7)

theorem expression_evaluation : 
  ∀ (a b c : ℚ), c = b - 11 → b = a + 3 → a = 5 → 
  (a + 2) ≠ 0 → (b - 3) ≠ 0 → (c + 7) ≠ 0 → 
  evaluate_expression a b c = 72 / 35 :=
by
  intros a b c hc hb ha h1 h2 h3
  rw [ha, hb, hc, evaluate_expression]
  -- The proof is not required.
  sorry

end expression_evaluation_l1195_119520


namespace problem_value_l1195_119503

theorem problem_value :
  1 - (-2) - 3 - (-4) - 5 - (-6) = 5 :=
by sorry

end problem_value_l1195_119503


namespace meal_combinations_count_l1195_119556

/-- Define the number of menu items -/
def num_menu_items : ℕ := 15

/-- Define the number of distinct combinations of meals Maryam and Jorge can order,
    considering they may choose the same dish and distinguishing who orders what -/
theorem meal_combinations_count (maryam_dishes jorge_dishes : ℕ) : 
  maryam_dishes = num_menu_items ∧ jorge_dishes = num_menu_items → 
  maryam_dishes * jorge_dishes = 225 :=
by
  intros h
  simp only [num_menu_items] at h -- Utilize the definition of num_menu_items
  sorry

end meal_combinations_count_l1195_119556


namespace find_k_and_slope_l1195_119524

theorem find_k_and_slope : 
  ∃ k : ℝ, (∃ y : ℝ, (3 + y = 8) ∧ (k = -3 * 3 + y)) ∧ (k = -4) ∧ 
  (∀ x y : ℝ, (x + y = 8) → (∃ m b : ℝ, y = m * x + b ∧ m = -1)) :=
by {
  sorry
}

end find_k_and_slope_l1195_119524


namespace find_number_l1195_119596

theorem find_number (x : ℝ) (h : (x - 8 - 12) / 5 = 7) : x = 55 :=
sorry

end find_number_l1195_119596


namespace max_tries_needed_to_open_lock_l1195_119568

-- Definitions and conditions
def num_buttons : ℕ := 9
def sequence_length : ℕ := 4
def opposite_trigrams : ℕ := 2  -- assumption based on the problem's example
def total_combinations : ℕ := 3024

theorem max_tries_needed_to_open_lock :
  (total_combinations - (8 * 1 * 7 * 6 + 8 * 6 * 1 * 6 + 8 * 6 * 4 * 1)) = 2208 :=
by
  sorry

end max_tries_needed_to_open_lock_l1195_119568


namespace volume_not_determined_l1195_119599

noncomputable def tetrahedron_volume_not_unique 
  (area1 area2 area3 : ℝ) (circumradius : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∃ a' b' c', 
      (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧ 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)))

theorem volume_not_determined 
  (area1 area2 area3 circumradius: ℝ) 
  (h: tetrahedron_volume_not_unique area1 area2 area3 circumradius) : 
  ¬ ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∀ a' b' c', 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)) → 
      (a = a' ∧ b = b' ∧ c = c')) := 
by sorry

end volume_not_determined_l1195_119599


namespace clock_hands_overlap_l1195_119574

theorem clock_hands_overlap:
  ∃ x y: ℚ,
  -- Conditions
  (60 * 10 + x = 60 * 11 * 54 + 6 / 11) ∧
  (y - (5 / 60) * y = 60) ∧
  (65 * 5 / 11 = y) := sorry

end clock_hands_overlap_l1195_119574


namespace all_positive_rationals_are_red_l1195_119517

-- Define the property of being red for rational numbers
def is_red (x : ℚ) : Prop :=
  ∃ n : ℕ, ∃ (f : ℕ → ℚ), f 0 = 1 ∧ (∀ m : ℕ, f (m + 1) = f m + 1 ∨ f (m + 1) = f m / (f m + 1)) ∧ f n = x

-- Proposition stating that all positive rational numbers are red
theorem all_positive_rationals_are_red :
  ∀ x : ℚ, 0 < x → is_red x :=
  by sorry

end all_positive_rationals_are_red_l1195_119517


namespace carousel_seat_count_l1195_119585

theorem carousel_seat_count
  (total_seats : ℕ)
  (colors : ℕ → Prop)
  (num_yellow num_blue num_red : ℕ)
  (num_colors : ∀ n, colors n → n = num_yellow ∨ n = num_blue ∨ n = num_red)
  (opposite_blue_red_7_3 : ∀ n, n = 7 ↔ n + 50 = 3)
  (opposite_yellow_red_7_23 : ∀ n, n = 7 ↔ n + 50 = 23)
  (total := 100)
 :
 (num_yellow = 34 ∧ num_blue = 20 ∧ num_red = 46) :=
by
  sorry

end carousel_seat_count_l1195_119585


namespace find_k_point_verification_l1195_119546

-- Definition of the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Condition that the point (2, 7) lies on the graph of the linear function
def passes_through (k : ℝ) : Prop := linear_function k 2 = 7

-- The actual proof task to verify the value of k
theorem find_k : ∃ k : ℝ, passes_through k ∧ k = 2 :=
by
  sorry

-- The condition that the point (-2, 1) is not on the graph with k = 2
def point_not_on_graph : Prop := ¬ (linear_function 2 (-2) = 1)

-- The actual proof task to verify the point (-2, 1) is not on the graph of y = 2x + 3
theorem point_verification : point_not_on_graph :=
by
  sorry

end find_k_point_verification_l1195_119546


namespace fractional_inspection_l1195_119519

theorem fractional_inspection:
  ∃ (J E A : ℝ),
  J + E + A = 1 ∧
  0.005 * J + 0.007 * E + 0.012 * A = 0.01 :=
by
  sorry

end fractional_inspection_l1195_119519


namespace largest_n_satisfying_ineq_l1195_119523
  
theorem largest_n_satisfying_ineq : ∃ n : ℕ, (n < 10) ∧ ∀ m : ℕ, (m < 10) → m ≤ n ∧ (n < 10) ∧ (m < 10) → n = 9 :=
by
  sorry

end largest_n_satisfying_ineq_l1195_119523


namespace fuel_consumption_per_100_km_l1195_119588

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end fuel_consumption_per_100_km_l1195_119588


namespace part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l1195_119581

noncomputable def A : Set ℝ := { x : ℝ | 3 < x ∧ x < 10 }
noncomputable def B : Set ℝ := { x : ℝ | x^2 - 9 * x + 14 < 0 }
noncomputable def C (m : ℝ) : Set ℝ := { x : ℝ | 5 - m < x ∧ x < 2 * m }

theorem part_I_A_inter_B : A ∩ B = { x : ℝ | 3 < x ∧ x < 7 } :=
sorry

theorem part_I_complement_A_union_B :
  (Aᶜ) ∪ B = { x : ℝ | x < 7 ∨ x ≥ 10 } :=
sorry

theorem part_II_range_of_m :
  {m : ℝ | C m ⊆ A ∩ B} = {m : ℝ | m ≤ 2} :=
sorry

end part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l1195_119581


namespace amount_added_to_doubled_number_l1195_119572

theorem amount_added_to_doubled_number (N A : ℝ) (h1 : N = 6.0) (h2 : 2 * N + A = 17) : A = 5.0 :=
by
  sorry

end amount_added_to_doubled_number_l1195_119572


namespace christel_gave_andrena_l1195_119550

theorem christel_gave_andrena (d m c a: ℕ) (h1: d = 20 - 2) (h2: c = 24) 
  (h3: a = c + 2) (h4: a = d + 3) : (24 - c = 5) :=
by { sorry }

end christel_gave_andrena_l1195_119550


namespace machine_c_more_bottles_l1195_119573

theorem machine_c_more_bottles (A B C : ℕ) 
  (hA : A = 12)
  (hB : B = A - 2)
  (h_total : 10 * A + 10 * B + 10 * C = 370) :
  C - B = 5 :=
by
  sorry

end machine_c_more_bottles_l1195_119573


namespace jeremy_sticker_distribution_l1195_119516

def number_of_ways_to_distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  (Nat.choose (total_stickers - 1) (sheets - 1))

theorem jeremy_sticker_distribution : number_of_ways_to_distribute_stickers 10 3 = 36 :=
by
  sorry

end jeremy_sticker_distribution_l1195_119516


namespace scientific_notation_l1195_119539

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end scientific_notation_l1195_119539


namespace rancher_loss_l1195_119542

-- Define the necessary conditions
def initial_head_of_cattle := 340
def original_total_price := 204000
def cattle_died := 172
def price_reduction_per_head := 150

-- Define the original and new prices per head
def original_price_per_head := original_total_price / initial_head_of_cattle
def new_price_per_head := original_price_per_head - price_reduction_per_head

-- Define the number of remaining cattle
def remaining_cattle := initial_head_of_cattle - cattle_died

-- Define the total amount at the new price
def total_amount_new_price := new_price_per_head * remaining_cattle

-- Define the loss
def loss := original_total_price - total_amount_new_price

-- Prove that the loss is $128,400
theorem rancher_loss : loss = 128400 := by
  sorry

end rancher_loss_l1195_119542


namespace expression_simplification_l1195_119548

noncomputable def given_expression : ℝ :=
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1)))

noncomputable def expected_expression : ℝ :=
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3)

theorem expression_simplification :
  given_expression = expected_expression :=
by
  sorry

end expression_simplification_l1195_119548


namespace chewing_gums_count_l1195_119540

-- Given conditions
def num_chocolate_bars : ℕ := 55
def num_candies : ℕ := 40
def total_treats : ℕ := 155

-- Definition to be proven
def num_chewing_gums : ℕ := total_treats - (num_chocolate_bars + num_candies)

-- Theorem statement
theorem chewing_gums_count : num_chewing_gums = 60 :=
by 
  -- here would be the proof steps, but it's omitted as per the instruction
  sorry

end chewing_gums_count_l1195_119540


namespace intersection_point_x_coordinate_l1195_119554

noncomputable def hyperbola (x y b : ℝ) := x^2 - (y^2 / b^2) = 1

noncomputable def c := 1 + Real.sqrt 3

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_point_x_coordinate
  (x y b : ℝ)
  (h_hyperbola : hyperbola x y b)
  (h_distance_foci : distance (2 * c, 0) (0, 0) = 2 * c)
  (h_circle_center : distance (x, y) (0, 0) = c)
  (h_p_distance : distance (x, y) (2 * c, 0) = c + 2) :
  x = (Real.sqrt 3 + 1) / 2 :=
sorry

end intersection_point_x_coordinate_l1195_119554


namespace sum_nonnegative_reals_l1195_119537

variable {x y z : ℝ}

theorem sum_nonnegative_reals (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 := 
by sorry

end sum_nonnegative_reals_l1195_119537


namespace students_drawn_from_A_l1195_119590

-- Define the conditions as variables (number of students in each school)
def studentsA := 3600
def studentsB := 5400
def studentsC := 1800
def sampleSize := 90

-- Define the total number of students
def totalStudents := studentsA + studentsB + studentsC

-- Define the proportion of students in School A
def proportionA := studentsA / totalStudents

-- Define the number of students to be drawn from School A using stratified sampling
def drawnFromA := sampleSize * proportionA

-- The theorem to prove
theorem students_drawn_from_A : drawnFromA = 30 :=
by
  sorry

end students_drawn_from_A_l1195_119590


namespace max_distance_between_sparkling_points_l1195_119533

theorem max_distance_between_sparkling_points (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 = 1) (h₂ : a₂^2 + b₂^2 = 1) :
  ∃ d, d = 2 ∧ ∀ (x y : ℝ), x = a₂ - a₁ ∧ y = b₂ - b₁ → (x ^ 2 + y ^ 2 = d ^ 2) :=
by
  sorry

end max_distance_between_sparkling_points_l1195_119533


namespace g_analytical_expression_g_minimum_value_l1195_119526

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1
noncomputable def M (a : ℝ) : ℝ := if (a ≥ 1/3 ∧ a ≤ 1/2) then f a 1 else f a 3
noncomputable def N (a : ℝ) : ℝ := f a (1/a)
noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 1/3 ∧ a ≤ 1/2 then M a - N a 
  else if a > 1/2 ∧ a ≤ 1 then M a - N a
  else 0 -- outside the given interval, by definition may be kept as 0

theorem g_analytical_expression (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) : 
  g a = if (1/3 ≤ a ∧ a ≤ 1/2) then a + 1/a - 2 else 9 * a + 1/a - 6 := 
sorry

theorem g_minimum_value (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ∃ (a' : ℝ), 1/3 ≤ a' ∧ a' ≤ 1 ∧ (∀ a, 1/3 ≤ a ∧ a ≤ 1 → g a ≥ g a') ∧ g a' = 1/2 := 
sorry

end g_analytical_expression_g_minimum_value_l1195_119526


namespace track_circumference_is_720_l1195_119512

variable (P Q : Type) -- Define the types of P and Q, e.g., as points or runners.

noncomputable def circumference_of_the_track (C : ℝ) : Prop :=
  ∃ y : ℝ, 
  (∃ first_meeting_condition : Prop, first_meeting_condition = (150 = y - 150) ∧
  ∃ second_meeting_condition : Prop, second_meeting_condition = (2*y - 90 = y + 90) ∧
  C = 2 * y)

theorem track_circumference_is_720 :
  circumference_of_the_track 720 :=
by
  sorry

end track_circumference_is_720_l1195_119512


namespace probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l1195_119511

namespace Sprinters

def P_A : ℚ := 2 / 5
def P_B : ℚ := 3 / 4
def P_C : ℚ := 1 / 3

def P_all_qualified := P_A * P_B * P_C
def P_two_qualified := P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C
def P_at_least_one_qualified := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C)

theorem probability_all_qualified : P_all_qualified = 1 / 10 :=
by 
  -- proof here
  sorry

theorem probability_two_qualified : P_two_qualified = 23 / 60 :=
by 
  -- proof here
  sorry

theorem probability_at_least_one_qualified : P_at_least_one_qualified = 9 / 10 :=
by 
  -- proof here
  sorry

end Sprinters

end probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l1195_119511


namespace total_points_of_players_l1195_119587

variables (Samanta Mark Eric Daisy Jake : ℕ)
variables (h1 : Samanta = Mark + 8)
variables (h2 : Mark = 3 / 2 * Eric)
variables (h3 : Eric = 6)
variables (h4 : Daisy = 3 / 4 * (Samanta + Mark + Eric))
variables (h5 : Jake = Samanta - Eric)
 
theorem total_points_of_players :
  Samanta + Mark + Eric + Daisy + Jake = 67 :=
sorry

end total_points_of_players_l1195_119587


namespace equivalent_operation_l1195_119551

theorem equivalent_operation (x : ℚ) :
  (x * (2/3)) / (5/6) = x * (4/5) :=
by
  -- Normal proof steps might follow here
  sorry

end equivalent_operation_l1195_119551


namespace math_proof_problem_l1195_119535

open Set

noncomputable def alpha : ℝ := (3 - Real.sqrt 5) / 2

theorem math_proof_problem (α_pos : 0 < α) (α_lt_delta : α < alpha) :
  ∃ n p : ℕ, p > α * 2^n ∧ ∃ S T : Finset (Fin n) → Finset (Fin n), (∀ i j, (S i) ∩ (T j) ≠ ∅) :=
  sorry

end math_proof_problem_l1195_119535


namespace sequence_problem_l1195_119538

theorem sequence_problem (S : ℕ → ℚ) (a : ℕ → ℚ) (h : ∀ n, S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ 
  (∀ n : ℕ, n > 0 → a n = (2^n - 1) / 2^(n-1)) :=
by
  sorry

end sequence_problem_l1195_119538


namespace sequence_of_8_numbers_l1195_119509

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end sequence_of_8_numbers_l1195_119509


namespace triangle_sides_ratios_l1195_119521

theorem triangle_sides_ratios (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) :
  a / (b + c) = b / (a + c) + c / (a + b) :=
sorry

end triangle_sides_ratios_l1195_119521
