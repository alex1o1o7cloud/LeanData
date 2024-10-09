import Mathlib

namespace no_solution_system_iff_n_eq_neg_cbrt_four_l599_59987

variable (n : ℝ)

theorem no_solution_system_iff_n_eq_neg_cbrt_four :
    (∀ x y z : ℝ, ¬ (2 * n * x + 3 * y = 2 ∧ 3 * n * y + 4 * z = 3 ∧ 4 * x + 2 * n * z = 4)) ↔
    n = - (4 : ℝ)^(1/3) := 
by
  sorry

end no_solution_system_iff_n_eq_neg_cbrt_four_l599_59987


namespace mrs_franklin_initial_valentines_l599_59988

theorem mrs_franklin_initial_valentines (v g l : ℕ) (h1 : g = 42) (h2 : l = 16) (h3 : v = g + l) : v = 58 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end mrs_franklin_initial_valentines_l599_59988


namespace find_y_l599_59990

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - t) (h2 : y = 3 * t + 6) (h3 : x = -6) : y = 33 := by
  sorry

end find_y_l599_59990


namespace cricket_team_right_handed_players_l599_59967

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (non_throwers : ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_throwers : ℕ := throwers)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers)
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers = non_throwers / 3) :
  total_right_handed = 59 :=
by
  rw [h1, h2] at *
  -- The remaining parts of the proof here are omitted for brevity.
  sorry

end cricket_team_right_handed_players_l599_59967


namespace DogHeight_is_24_l599_59949

-- Define the given conditions as Lean definitions (variables and equations)
variable (CarterHeight DogHeight BettyHeight : ℝ)

-- Assume the conditions given in the problem
axiom h1 : CarterHeight = 2 * DogHeight
axiom h2 : BettyHeight + 12 = CarterHeight
axiom h3 : BettyHeight = 36

-- State the proposition (the height of Carter's dog)
theorem DogHeight_is_24 : DogHeight = 24 :=
by
  -- Proof goes here
  sorry

end DogHeight_is_24_l599_59949


namespace tom_average_speed_l599_59915

theorem tom_average_speed
  (total_distance : ℕ)
  (distance1 : ℕ)
  (speed1 : ℕ)
  (distance2 : ℕ)
  (speed2 : ℕ)
  (H : total_distance = distance1 + distance2)
  (H1 : distance1 = 12)
  (H2 : speed1 = 24)
  (H3 : distance2 = 48)
  (H4 : speed2 = 48) :
  (total_distance : ℚ) / ((distance1 : ℚ) / speed1 + (distance2 : ℚ) / speed2) = 40 :=
by
  sorry

end tom_average_speed_l599_59915


namespace equation_of_line_containing_chord_l599_59905

theorem equation_of_line_containing_chord (x y : ℝ) : 
  (y^2 = -8 * x) ∧ ((-1, 1) = ((x + x) / 2, (y + y) / 2)) →
  4 * x + y + 3 = 0 :=
by 
  sorry

end equation_of_line_containing_chord_l599_59905


namespace f_is_periodic_l599_59920

noncomputable def f (x : ℝ) : ℝ := x - ⌈x⌉

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x :=
by 
  intro x
  sorry

end f_is_periodic_l599_59920


namespace sufficient_not_necessary_l599_59916

def M : Set Int := {0, 1, 2}
def N : Set Int := {-1, 0, 1, 2}

theorem sufficient_not_necessary (a : Int) : a ∈ M → a ∈ N ∧ ¬(a ∈ N → a ∈ M) := by
  sorry

end sufficient_not_necessary_l599_59916


namespace desk_height_l599_59906

variables (h l w : ℝ)

theorem desk_height
  (h_eq_2l_50 : h + 2 * l = 50)
  (h_eq_2w_40 : h + 2 * w = 40)
  (l_minus_w_eq_5 : l - w = 5) :
  h = 30 :=
by {
  sorry
}

end desk_height_l599_59906


namespace base_length_of_triangle_l599_59979

theorem base_length_of_triangle (height area : ℕ) (h1 : height = 8) (h2 : area = 24) : 
  ∃ base : ℕ, (1/2 : ℚ) * base * height = area ∧ base = 6 := by
  sorry

end base_length_of_triangle_l599_59979


namespace intersection_correct_l599_59950

open Set

noncomputable def A := {x : ℕ | x^2 - x - 2 ≤ 0}
noncomputable def B := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def A_cap_B := A ∩ {x : ℕ | (x : ℝ) ∈ B}

theorem intersection_correct : A_cap_B = {0, 1} :=
sorry

end intersection_correct_l599_59950


namespace problem_statement_l599_59995

-- Proposition p: For any x ∈ ℝ, 2^x > x^2
def p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Proposition q: "ab > 4" is a sufficient but not necessary condition for "a > 2 and b > 2"
def q : Prop := (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4)) ∧ ¬ (∀ a b : ℝ, (a * b > 4) → (a > 2 ∧ b > 2))

-- Problem statement: Determine that the true statement is ¬p ∧ ¬q
theorem problem_statement : ¬p ∧ ¬q := by
  sorry

end problem_statement_l599_59995


namespace part_1_solution_part_2_solution_l599_59952

def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem part_1_solution (x : ℝ) : f x < 3 ↔ -4 / 3 < x ∧ x < 0 :=
by
  sorry

theorem part_2_solution (a : ℝ) : (∀ x, ¬ (f x < a)) → a ≤ 2 :=
by
  sorry

end part_1_solution_part_2_solution_l599_59952


namespace merchant_markup_percentage_l599_59909

theorem merchant_markup_percentage
  (CP : ℕ) (discount_percent : ℚ) (profit_percent : ℚ)
  (mp : ℚ := CP + x)
  (sp : ℚ := (1 - discount_percent) * mp)
  (final_sp : ℚ := CP * (1 + profit_percent)) :
  discount_percent = 15 / 100 ∧ profit_percent = 19 / 100 ∧ CP = 100 → 
  sp = 85 + 0.85 * x → 
  final_sp = 119 →
  x = 40 :=
by 
  sorry

end merchant_markup_percentage_l599_59909


namespace max_marks_obtainable_l599_59965

theorem max_marks_obtainable 
  (math_pass_percentage : ℝ := 36 / 100)
  (phys_pass_percentage : ℝ := 40 / 100)
  (chem_pass_percentage : ℝ := 45 / 100)
  (math_marks : ℕ := 130)
  (math_fail_margin : ℕ := 14)
  (phys_marks : ℕ := 120)
  (phys_fail_margin : ℕ := 20)
  (chem_marks : ℕ := 160)
  (chem_fail_margin : ℕ := 10) : 
  ∃ max_total_marks : ℤ, max_total_marks = 1127 := 
by 
  sorry  -- Proof not required

end max_marks_obtainable_l599_59965


namespace min_fraction_value_l599_59908

theorem min_fraction_value (x : ℝ) (hx : x > 9) : ∃ y, y = 36 ∧ (∀ z, z = (x^2 / (x - 9)) → y ≤ z) :=
by
  sorry

end min_fraction_value_l599_59908


namespace sum_of_roots_l599_59947

-- Given condition: the equation to be solved
def equation (x : ℝ) : Prop :=
  x^2 - 7 * x + 2 = 16

-- Define the proof problem: the sum of the values of x that satisfy the equation
theorem sum_of_roots : 
  (∀ x : ℝ, equation x) → x^2 - 7*x - 14 = 0 → (root : ℝ) → (root₀ + root₁) = 7 :=
by
  sorry

end sum_of_roots_l599_59947


namespace Vins_total_miles_l599_59998

theorem Vins_total_miles : 
  let dist_library_one_way := 6
  let dist_school_one_way := 5
  let dist_friend_one_way := 8
  let extra_miles := 1
  let shortcut_miles := 2
  let days_per_week := 7
  let weeks := 4

  -- Calculate weekly miles
  let library_round_trip := (dist_library_one_way + dist_library_one_way + extra_miles)
  let total_library_weekly := library_round_trip * 3

  let school_round_trip := (dist_school_one_way + dist_school_one_way + extra_miles)
  let total_school_weekly := school_round_trip * 2

  let friend_round_trip := dist_friend_one_way + (dist_friend_one_way - shortcut_miles)
  let total_friend_weekly := friend_round_trip / 2 -- Every two weeks

  let total_weekly := total_library_weekly + total_school_weekly + total_friend_weekly

  -- Calculate total miles over the weeks
  let total_miles := total_weekly * weeks

  total_miles = 272 := sorry

end Vins_total_miles_l599_59998


namespace temperature_on_Friday_l599_59972

variable {M T W Th F : ℝ}

theorem temperature_on_Friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (hM : M = 41) :
  F = 33 :=
by
  -- Proof goes here
  sorry

end temperature_on_Friday_l599_59972


namespace simplify_and_evaluate_division_l599_59984

theorem simplify_and_evaluate_division (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a ^ 3 / (a ^ 2 - 4 * a + 4)) = 1 / 3 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_division_l599_59984


namespace rectangles_in_cube_l599_59974

/-- Number of rectangles that can be formed by the vertices of a cube is 12. -/
theorem rectangles_in_cube : 
  ∃ (n : ℕ), (n = 12) := by
  -- The cube has vertices, and squares are a subset of rectangles.
  -- We need to count rectangles including squares among vertices of the cube.
  sorry

end rectangles_in_cube_l599_59974


namespace pythagorean_triple_solution_l599_59970

theorem pythagorean_triple_solution
  (x y z a b : ℕ)
  (h1 : x^2 + y^2 = z^2)
  (h2 : Nat.gcd x y = 1)
  (h3 : 2 ∣ y)
  (h4 : a > b)
  (h5 : b > 0)
  (h6 : (Nat.gcd a b = 1))
  (h7 : ((a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))) 
  : (x = a^2 - b^2 ∧ y = 2 * a * b ∧ z = a^2 + b^2) := 
sorry

end pythagorean_triple_solution_l599_59970


namespace solve_equation_l599_59912

variable (x : ℝ)

theorem solve_equation (h : x * (x - 4) = x - 6) : x = 2 ∨ x = 3 := 
sorry

end solve_equation_l599_59912


namespace james_new_friends_l599_59997

-- Definitions and assumptions based on the conditions provided
def initial_friends := 20
def lost_friends := 2
def friends_after_loss : ℕ := initial_friends - lost_friends
def friends_upon_arrival := 19

-- Definition of new friends made
def new_friends : ℕ := friends_upon_arrival - friends_after_loss

-- Statement to prove
theorem james_new_friends :
  new_friends = 1 :=
by
  -- Solution proof would be inserted here
  sorry

end james_new_friends_l599_59997


namespace mixed_gender_selection_count_is_correct_l599_59924

/- Define the given constants -/
def num_male_students : ℕ := 5
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students
def selection_size : ℕ := 3

/- Define the function to compute binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

/- The Lean 4 statement -/
theorem mixed_gender_selection_count_is_correct
  (num_male_students num_female_students total_students selection_size : ℕ)
  (hc1 : num_male_students = 5)
  (hc2 : num_female_students = 3)
  (hc3 : total_students = num_male_students + num_female_students)
  (hc4 : selection_size = 3) :
  binom total_students selection_size 
  - binom num_male_students selection_size
  - binom num_female_students selection_size = 45 := 
  by 
    -- Only the statement is required
    sorry

end mixed_gender_selection_count_is_correct_l599_59924


namespace geometric_sequence_a5_value_l599_59907

theorem geometric_sequence_a5_value :
  ∃ (a : ℕ → ℝ) (r : ℝ), (a 3)^2 - 4 * a 3 + 3 = 0 ∧ 
                         (a 7)^2 - 4 * a 7 + 3 = 0 ∧ 
                         (a 3) * (a 7) = 3 ∧ 
                         (a 3) + (a 7) = 4 ∧ 
                         a 5 = (a 3 * a 7).sqrt :=
sorry

end geometric_sequence_a5_value_l599_59907


namespace four_digit_numbers_with_property_l599_59978

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l599_59978


namespace fraction_sum_is_half_l599_59937

theorem fraction_sum_is_half :
  (1/5 : ℚ) + (3/10 : ℚ) = 1/2 :=
by linarith

end fraction_sum_is_half_l599_59937


namespace typing_pages_l599_59901

theorem typing_pages (typists : ℕ) (pages min : ℕ) 
  (h_typists_can_type_two_pages_in_two_minutes : typists * 2 / min = pages / min) 
  (h_10_typists_type_25_pages_in_5_minutes : 10 * 25 / 5 = pages / min) :
  pages / min = 2 := 
sorry

end typing_pages_l599_59901


namespace sequence_properties_l599_59971

theorem sequence_properties
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 = 5)
  (h2 : ∀ n, n % 2 = 1 → a (n + 1) - a n = 1)
  (h3 : ∀ n, n % 2 = 0 → a (n + 1) - a n = 3) :
  (a 1 = 2) ∧ (a 2 = 3) ∧
  (∀ n, a (2 * n - 1) = 2 * (2 * n - 1)) ∧
  (∀ n, a (2 * n) = 2 * 2 * n - 1) :=
by
  sorry

end sequence_properties_l599_59971


namespace minimum_BC_length_l599_59999

theorem minimum_BC_length (AB AC DC BD BC : ℕ)
  (h₁ : AB = 5) (h₂ : AC = 12) (h₃ : DC = 8) (h₄ : BD = 20) (h₅ : BC > 12) : BC = 13 :=
by
  sorry

end minimum_BC_length_l599_59999


namespace total_shirts_sold_l599_59948

theorem total_shirts_sold (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 : ℕ) (h1 : p1 = 20) (h2 : p2 = 22) (h3 : p3 = 25)
(h4 : p4 + p5 + p6 + p7 + p8 + p9 + p10 = 133) (h5 : ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10) / 10) > 20)
: p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 = 200 ∧ 10 = 10 := sorry

end total_shirts_sold_l599_59948


namespace minimum_value_problem1_minimum_value_problem2_l599_59961

theorem minimum_value_problem1 (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y >= 6 := 
sorry

theorem minimum_value_problem2 (x : ℝ) (h : x > 1) : 
  ∃ y, y = (x^2 + 8) / (x - 1) ∧ y >= 8 := 
sorry

end minimum_value_problem1_minimum_value_problem2_l599_59961


namespace juju_juice_bar_l599_59951

theorem juju_juice_bar (M P : ℕ) 
  (h₁ : 6 * P = 54)
  (h₂ : 5 * M + 6 * P = 94) : 
  M + P = 17 := 
sorry

end juju_juice_bar_l599_59951


namespace find_TU2_l599_59993

-- Define the structure of the square, distances, and points
structure square (P Q R S T U : Type) :=
(PQ : ℝ)
(PT QU QT RU TU2 : ℝ)
(h1 : PQ = 15)
(h2 : PT = 7)
(h3 : QU = 7)
(h4 : QT = 17)
(h5 : RU = 17)
(h6 : TU2 = TU^2)
(h7 : TU2 = 1073)

-- The main proof statement
theorem find_TU2 {P Q R S T U : Type} (sq : square P Q R S T U) : sq.TU2 = 1073 := by
  sorry

end find_TU2_l599_59993


namespace compare_fractions_l599_59944

variable {a b c d : ℝ}

theorem compare_fractions (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  (b / (a - c)) < (a / (b - d)) := 
by
  sorry

end compare_fractions_l599_59944


namespace area_of_DEF_l599_59960

variable (t4_area t5_area t6_area : ℝ) (a_DEF : ℝ)

def similar_triangles_area := (t4_area = 1) ∧ (t5_area = 16) ∧ (t6_area = 36)

theorem area_of_DEF 
  (h : similar_triangles_area t4_area t5_area t6_area) :
  a_DEF = 121 := sorry

end area_of_DEF_l599_59960


namespace golden_apples_per_pint_l599_59936

-- Data definitions based on given conditions and question
def farmhands : ℕ := 6
def apples_per_hour : ℕ := 240
def hours : ℕ := 5
def ratio_golden_to_pink : ℕ × ℕ := (1, 2)
def pints_of_cider : ℕ := 120
def pink_lady_per_pint : ℕ := 40

-- Total apples picked by farmhands in 5 hours
def total_apples_picked : ℕ := farmhands * apples_per_hour * hours

-- Total pink lady apples picked
def total_pink_lady_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.2) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total golden delicious apples picked
def total_golden_delicious_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.1) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total pink lady apples used for 120 pints of cider
def pink_lady_apples_used : ℕ := pints_of_cider * pink_lady_per_pint

-- Number of golden delicious apples used per pint of cider
def golden_delicious_apples_per_pint : ℕ := total_golden_delicious_apples / pints_of_cider

-- Main theorem to prove
theorem golden_apples_per_pint : golden_delicious_apples_per_pint = 20 := by
  -- Start proof (proof body is omitted)
  sorry

end golden_apples_per_pint_l599_59936


namespace distribute_tourists_l599_59927

-- Define the number of ways k tourists can distribute among n cinemas
def num_ways (n k : ℕ) : ℕ := n^k

-- Theorem stating the number of distribution ways
theorem distribute_tourists (n k : ℕ) : num_ways n k = n^k :=
by sorry

end distribute_tourists_l599_59927


namespace range_of_m_l599_59942

noncomputable def quadraticExpr (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + 4 * m * x + m + 3

theorem range_of_m :
  (∀ x : ℝ, quadraticExpr m x ≥ 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l599_59942


namespace gcd_g50_g52_l599_59962

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 2 * x^2 + x + 2023

-- Define the integers n1 and n2 corresponding to g(50) and g(52)
def n1 : ℤ := g 50
def n2 : ℤ := g 52

-- Statement of the proof goal
theorem gcd_g50_g52 : Int.gcd n1 n2 = 1 := by
  sorry

end gcd_g50_g52_l599_59962


namespace crayons_slightly_used_l599_59977

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end crayons_slightly_used_l599_59977


namespace torn_pages_count_l599_59921

theorem torn_pages_count (pages : Finset ℕ) (h1 : ∀ p ∈ pages, 1 ≤ p ∧ p ≤ 100) (h2 : pages.sum id = 4949) : 
  100 - pages.card = 3 := 
by
  sorry

end torn_pages_count_l599_59921


namespace simplify_and_rationalize_l599_59986

theorem simplify_and_rationalize :
  ( (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 9 / Real.sqrt 13) = 
    (3 * Real.sqrt 15015) / 1001 ) :=
by
  sorry

end simplify_and_rationalize_l599_59986


namespace bacterium_probability_l599_59959

noncomputable def probability_bacterium_in_small_cup
  (total_volume : ℚ) (small_cup_volume : ℚ) (contains_bacterium : Bool) : ℚ :=
if contains_bacterium then small_cup_volume / total_volume else 0

theorem bacterium_probability
  (total_volume : ℚ) (small_cup_volume : ℚ) (bacterium_present : Bool) :
  total_volume = 2 ∧ small_cup_volume = 0.1 ∧ bacterium_present = true →
  probability_bacterium_in_small_cup 2 0.1 true = 0.05 :=
by
  intros h
  sorry

end bacterium_probability_l599_59959


namespace quadr_pyramid_edge_sum_is_36_l599_59926

def sum_edges_quad_pyr (hex_sum_edges : ℕ) (hex_num_edges : ℕ) (quad_num_edges : ℕ) : ℕ :=
  let length_one_edge := hex_sum_edges / hex_num_edges
  length_one_edge * quad_num_edges

theorem quadr_pyramid_edge_sum_is_36 :
  sum_edges_quad_pyr 81 18 8 = 36 :=
by
  -- We defer proof
  sorry

end quadr_pyramid_edge_sum_is_36_l599_59926


namespace integral_evaluation_l599_59903

noncomputable def definite_integral (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem integral_evaluation : 
  definite_integral 1 2 (fun x => 1 / x + x) = Real.log 2 + 3 / 2 :=
  sorry

end integral_evaluation_l599_59903


namespace sample_size_calculation_l599_59996

theorem sample_size_calculation 
    (total_teachers : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) 
    (sample_size_female_students : ℕ) 
    (H1 : total_teachers = 100) (H2 : total_male_students = 600) 
    (H3 : total_female_students = 500) (H4 : sample_size_female_students = 40)
    : (sample_size_female_students * (total_teachers + total_male_students + total_female_students) / total_female_students) = 96 := 
by
  /- sorry, proof omitted -/
  sorry
  
end sample_size_calculation_l599_59996


namespace icing_cubes_count_31_l599_59958

def cake_cubed (n : ℕ) := n^3

noncomputable def slabs_with_icing (n : ℕ): ℕ := 
    let num_faces := 3
    let edge_per_face := n - 1
    let edges_with_icing := num_faces * edge_per_face * (n - 2)
    edges_with_icing + (n - 2) * 4 * (n - 2)

theorem icing_cubes_count_31 : ∀ (n : ℕ), n = 5 → slabs_with_icing n = 31 :=
by
  intros n hn
  revert hn
  sorry

end icing_cubes_count_31_l599_59958


namespace find_g_neg6_l599_59945

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l599_59945


namespace area_of_PINE_l599_59940

def PI := 6
def IN := 15
def NE := 6
def EP := 25
def sum_angles := 60 

theorem area_of_PINE : 
  (∃ (area : ℝ), area = (100 * Real.sqrt 3) / 3) := 
sorry

end area_of_PINE_l599_59940


namespace monotonically_increasing_interval_l599_59953

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotonically_increasing_interval : 
  ∃ (a b : ℝ), a = -Real.pi / 3 ∧ b = Real.pi / 6 ∧ ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y :=
by
  sorry

end monotonically_increasing_interval_l599_59953


namespace min_x_plus_3y_l599_59925

noncomputable def minimum_x_plus_3y (x y : ℝ) : ℝ :=
  if h : (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) then x + 3*y else 0

theorem min_x_plus_3y : ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) → x + 3*y = 6 :=
by
  intros x y h
  sorry

end min_x_plus_3y_l599_59925


namespace range_of_a_l599_59975

variable {x a : ℝ}

def p (x : ℝ) := x^2 - 8 * x - 20 > 0
def q (a : ℝ) (x : ℝ) := x^2 - 2 * x + 1 - a^2 > 0

theorem range_of_a (h₀ : ∀ x, p x → q a x) (h₁ : a > 0) : 0 < a ∧ a ≤ 3 := 
by 
  sorry

end range_of_a_l599_59975


namespace distinct_painted_cubes_l599_59928

-- Define the context of the problem
def num_faces : ℕ := 6

def total_paintings : ℕ := num_faces.factorial

def num_rotations : ℕ := 24

-- Statement of the theorem
theorem distinct_painted_cubes (h1 : total_paintings = 720) (h2 : num_rotations = 24) : 
  total_paintings / num_rotations = 30 := by
  sorry

end distinct_painted_cubes_l599_59928


namespace usable_field_area_l599_59933

open Float

def breadth_of_field (P : ℕ) (extra_length : ℕ) := (P / 2 - extra_length) / 2

def length_of_field (b : ℕ) (extra_length : ℕ) := b + extra_length

def effective_length (l : ℕ) (obstacle_length : ℕ) := l - obstacle_length

def effective_breadth (b : ℕ) (obstacle_breadth : ℕ) := b - obstacle_breadth

def field_area (length : ℕ) (breadth : ℕ) := length * breadth 

theorem usable_field_area : 
  ∀ (P extra_length obstacle_length obstacle_breadth : ℕ), 
  P = 540 -> extra_length = 30 -> obstacle_length = 10 -> obstacle_breadth = 5 -> 
  field_area (effective_length (length_of_field (breadth_of_field P extra_length) extra_length) obstacle_length) (effective_breadth (breadth_of_field P extra_length) obstacle_breadth) = 16100 := by
  sorry

end usable_field_area_l599_59933


namespace calculate_value_l599_59911

def f (x : ℕ) : ℕ := 2 * x - 3
def g (x : ℕ) : ℕ := x^2 + 1

theorem calculate_value : f (1 + g 3) = 19 := by
  sorry

end calculate_value_l599_59911


namespace remainder_of_addition_and_division_l599_59902

theorem remainder_of_addition_and_division :
  (3452179 + 50) % 7 = 4 :=
by
  sorry

end remainder_of_addition_and_division_l599_59902


namespace total_miles_driven_l599_59913

-- Given constants and conditions
def city_mpg : ℝ := 30
def highway_mpg : ℝ := 37
def total_gallons : ℝ := 11
def highway_extra_miles : ℕ := 5

-- Variable for the number of city miles
variable (x : ℝ)

-- Conditions encapsulated in a theorem statement
theorem total_miles_driven:
  (x / city_mpg) + ((x + highway_extra_miles) / highway_mpg) = total_gallons →
  x + (x + highway_extra_miles) = 365 :=
by
  sorry

end total_miles_driven_l599_59913


namespace specific_value_correct_l599_59917

noncomputable def specific_value (x : ℝ) : ℝ :=
  (3 / 5) * (x ^ 2)

theorem specific_value_correct :
  specific_value 14.500000000000002 = 126.15000000000002 :=
by
  sorry

end specific_value_correct_l599_59917


namespace joan_gave_mike_seashells_l599_59983

-- Definitions based on the conditions
def original_seashells : ℕ := 79
def remaining_seashells : ℕ := 16
def given_seashells := original_seashells - remaining_seashells

-- The theorem we want to prove
theorem joan_gave_mike_seashells : given_seashells = 63 := by
  sorry

end joan_gave_mike_seashells_l599_59983


namespace proof_inequality_l599_59966

theorem proof_inequality (p q r : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hpr_lt_qr : p * r < q * r) : 
  p < q :=
by 
  sorry

end proof_inequality_l599_59966


namespace coefficient_a5_l599_59955

theorem coefficient_a5 (a a1 a2 a3 a4 a5 a6 : ℝ) (h :  (∀ x : ℝ, x^6 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) :
  a5 = 6 :=
sorry

end coefficient_a5_l599_59955


namespace ratio_of_areas_l599_59994

theorem ratio_of_areas (C1 C2 : ℝ) (h : (60 / 360) * C1 = (30 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l599_59994


namespace handshakes_7_boys_l599_59946

theorem handshakes_7_boys : Nat.choose 7 2 = 21 :=
by
  sorry

end handshakes_7_boys_l599_59946


namespace problem_prime_square_plus_two_l599_59929

theorem problem_prime_square_plus_two (P : ℕ) (hP_prime : Prime P) (hP2_plus_2_prime : Prime (P^2 + 2)) : P^4 + 1921 = 2002 :=
by
  sorry

end problem_prime_square_plus_two_l599_59929


namespace train_arrival_time_l599_59939

-- Define the time type
structure Time where
  hour : Nat
  minute : Nat

namespace Time

-- Define the addition of minutes to a time.
def add_minutes (t : Time) (m : Nat) : Time :=
  let new_minutes := t.minute + m
  if new_minutes < 60 then 
    { hour := t.hour, minute := new_minutes }
  else 
    { hour := t.hour + new_minutes / 60, minute := new_minutes % 60 }

-- Define the departure time
def departure_time : Time := { hour := 9, minute := 45 }

-- Define the travel time in minutes
def travel_time : Nat := 15

-- Define the expected arrival time
def expected_arrival_time : Time := { hour := 10, minute := 0 }

-- The theorem we need to prove
theorem train_arrival_time:
  add_minutes departure_time travel_time = expected_arrival_time := by
  sorry

end train_arrival_time_l599_59939


namespace find_a_find_m_l599_59930

-- Definition of the odd function condition
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- The first proof problem
theorem find_a (a : ℝ) (h_odd : odd_function (fun x => Real.log (Real.exp x + a + 1))) : a = -1 :=
sorry

-- Definitions of the two functions involved in the second proof problem
noncomputable def f1 (x : ℝ) : ℝ :=
if x = 0 then 0 else Real.log x / x

noncomputable def f2 (x m : ℝ) : ℝ :=
x^2 - 2 * Real.exp 1 * x + m

-- The second proof problem
theorem find_m (m : ℝ) (h_root : ∃! x, f1 x = f2 x m) : m = Real.exp 2 + 1 / Real.exp 1 :=
sorry

end find_a_find_m_l599_59930


namespace evaluate_star_l599_59943

-- Define the operation c star d
def star (c d : ℤ) : ℤ := c^2 - 2 * c * d + d^2

-- State the theorem to prove the given problem
theorem evaluate_star : (star 3 5) = 4 := by
  sorry

end evaluate_star_l599_59943


namespace distance_per_interval_l599_59981

-- Definitions for the conditions
def total_distance : ℕ := 3  -- miles
def total_time : ℕ := 45  -- minutes
def interval_time : ℕ := 15  -- minutes per interval

-- Mathematical problem statement
theorem distance_per_interval :
  (total_distance / (total_time / interval_time) = 1) :=
by 
  sorry

end distance_per_interval_l599_59981


namespace repaired_shoes_last_correct_l599_59931

noncomputable def repaired_shoes_last := 
  let repair_cost: ℝ := 10.50
  let new_shoes_cost: ℝ := 30.00
  let new_shoes_years: ℝ := 2.0
  let percentage_increase: ℝ := 42.857142857142854 / 100
  (T : ℝ) -> 15.00 = (repair_cost / T) * (1 + percentage_increase) → T = 1

theorem repaired_shoes_last_correct : repaired_shoes_last :=
by
  sorry

end repaired_shoes_last_correct_l599_59931


namespace ice_cream_initial_amount_l599_59991

noncomputable def initial_ice_cream (milkshake_count : ℕ) : ℕ :=
  12 * milkshake_count

theorem ice_cream_initial_amount (m_i m_f : ℕ) (milkshake_count : ℕ) (I_f : ℕ) :
  m_i = 72 →
  m_f = 8 →
  milkshake_count = (m_i - m_f) / 4 →
  I_f = initial_ice_cream milkshake_count →
  I_f = 192 :=
by
  intros hmi hmf hcount hIc
  sorry

end ice_cream_initial_amount_l599_59991


namespace find_integer_for_prime_l599_59932

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem find_integer_for_prime (n : ℤ) :
  is_prime (4 * n^4 + 1) ↔ n = 1 :=
by
  sorry

end find_integer_for_prime_l599_59932


namespace lottery_prob_correct_l599_59973

def possibleMegaBalls : ℕ := 30
def possibleWinnerBalls : ℕ := 49
def drawnWinnerBalls : ℕ := 6

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def winningProbability : ℚ :=
  (1 : ℚ) / possibleMegaBalls * (1 : ℚ) / combination possibleWinnerBalls drawnWinnerBalls

theorem lottery_prob_correct :
  winningProbability = 1 / 419514480 := by
  sorry

end lottery_prob_correct_l599_59973


namespace boys_camp_problem_l599_59918

noncomputable def total_boys_in_camp : ℝ :=
  let schoolA_fraction := 0.20
  let science_fraction := 0.30
  let non_science_boys := 63
  let non_science_fraction := 1 - science_fraction
  let schoolA_boys := (non_science_boys / non_science_fraction)
  schoolA_boys / schoolA_fraction

theorem boys_camp_problem : total_boys_in_camp = 450 := 
by
  sorry

end boys_camp_problem_l599_59918


namespace hikers_rate_l599_59922

-- Define the conditions from the problem
variables (R : ℝ) (time_up time_down : ℝ) (distance_down : ℝ)

-- Conditions given in the problem
axiom condition1 : time_up = 2
axiom condition2 : time_down = 2
axiom condition3 : distance_down = 9
axiom condition4 : (distance_down / time_down) = 1.5 * R

-- The proof goal
theorem hikers_rate (h1 : time_up = 2) 
                    (h2 : time_down = 2) 
                    (h3 : distance_down = 9) 
                    (h4 : distance_down / time_down = 1.5 * R) : R = 3 := 
by 
  sorry

end hikers_rate_l599_59922


namespace triangle_acute_of_angles_sum_gt_90_l599_59976

theorem triangle_acute_of_angles_sum_gt_90 
  (α β γ : ℝ) 
  (h₁ : α + β + γ = 180) 
  (h₂ : α + β > 90) 
  (h₃ : α + γ > 90) 
  (h₄ : β + γ > 90) 
  : α < 90 ∧ β < 90 ∧ γ < 90 :=
sorry

end triangle_acute_of_angles_sum_gt_90_l599_59976


namespace find_m_l599_59969

theorem find_m (m : ℝ) : (∀ x > 0, x^2 - 2 * (m^2 + m + 1) * Real.log x ≥ 1) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end find_m_l599_59969


namespace charlie_ride_distance_l599_59900

-- Define the known values
def oscar_ride : ℝ := 0.75
def difference : ℝ := 0.5

-- Define Charlie's bus ride distance
def charlie_ride : ℝ := oscar_ride - difference

-- The theorem to be proven
theorem charlie_ride_distance : charlie_ride = 0.25 := 
by sorry

end charlie_ride_distance_l599_59900


namespace find_a4_b4_c4_l599_59992

variables {a b c : ℝ}

theorem find_a4_b4_c4 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 0.1) : a^4 + b^4 + c^4 = 0.005 :=
sorry

end find_a4_b4_c4_l599_59992


namespace people_per_bus_l599_59964

def num_vans : ℝ := 6.0
def num_buses : ℝ := 8.0
def people_per_van : ℝ := 6.0
def extra_people : ℝ := 108.0

theorem people_per_bus :
  let people_vans := num_vans * people_per_van
  let people_buses := people_vans + extra_people
  let people_per_bus := people_buses / num_buses
  people_per_bus = 18.0 :=
by 
  sorry

end people_per_bus_l599_59964


namespace tangent_line_at_P_l599_59923

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x

theorem tangent_line_at_P 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : P.1 + P.2 = 0)
  (h2 : f' P.1 a = -1) 
  (h3 : P.2 = f P.1 a) 
  : P = (1, -1) ∨ P = (-1, 1) := 
  sorry

end tangent_line_at_P_l599_59923


namespace correct_propositions_l599_59968

-- Definitions according to the given conditions
def generatrix_cylinder (p1 p2 : Point) (c : Cylinder) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def generatrix_cone (v : Point) (p : Point) (c : Cone) : Prop :=
  -- Check if the line from the vertex to a base point is a generatrix
  sorry

def generatrix_frustum (p1 p2 : Point) (f : Frustum) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def parallel_generatrices_cylinder (gen1 gen2 : Line) (c : Cylinder) : Prop :=
  -- Check if two generatrices of the cylinder are parallel
  sorry

-- The theorem stating propositions ② and ④ are correct
theorem correct_propositions :
  generatrix_cone vertex point cone ∧
  parallel_generatrices_cylinder gen1 gen2 cylinder :=
by
  sorry

end correct_propositions_l599_59968


namespace area_of_given_triangle_l599_59954

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def vertex_A : ℝ × ℝ := (-1, 4)
def vertex_B : ℝ × ℝ := (7, 0)
def vertex_C : ℝ × ℝ := (11, 5)

theorem area_of_given_triangle : area_of_triangle vertex_A vertex_B vertex_C = 28 :=
by
  show 1 / 2 * |(-1) * (0 - 5) + 7 * (5 - 4) + 11 * (4 - 0)| = 28
  sorry

end area_of_given_triangle_l599_59954


namespace problem_statement_l599_59941

noncomputable def ratio_AD_AB (AB AD : ℝ) (angle_A angle_B angle_ADE : ℝ) : Prop :=
  angle_A = 60 ∧ angle_B = 45 ∧ angle_ADE = 45 ∧
  AD / AB = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2)

theorem problem_statement {AB AD : ℝ} (angle_A angle_B angle_ADE : ℝ) 
  (h1 : angle_A = 60)
  (h2 : angle_B = 45)
  (h3 : angle_ADE = 45) : 
  ratio_AD_AB AB AD angle_A angle_B angle_ADE := by {
    sorry
}

end problem_statement_l599_59941


namespace prove_road_length_l599_59963

-- Define variables for days taken by team A, B, and C
variables {a b c : ℕ}

-- Define the daily completion rates for teams A, B, and C
def rateA : ℕ := 300
def rateB : ℕ := 240
def rateC : ℕ := 180

-- Define the maximum length of the road
def max_length : ℕ := 3500

-- Define the total section of the road that team A completes in a days
def total_A (a : ℕ) : ℕ := a * rateA

-- Define the total section of the road that team B completes in b days and 18 hours
def total_B (a b : ℕ) : ℕ := 240 * (a + b) + 180

-- Define the total section of the road that team C completes in c days and 8 hours
def total_C (a b c : ℕ) : ℕ := 180 * (a + b + c) + 60

-- Define the constraint on the sum of days taken: a + b + c
def total_days (a b c : ℕ) : ℕ := a + b + c

-- The proof goal: Prove that (a * 300 == 3300) given the conditions
theorem prove_road_length :
  (total_A a = 3300) ∧ (total_B a b ≤ max_length) ∧ (total_C a b c ≤ max_length) ∧ (total_days a b c ≤ 19) :=
sorry

end prove_road_length_l599_59963


namespace math_problem_l599_59956

variable (a b c d : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
variable (h1 : a^3 + b^3 + 3 * a * b = 1)
variable (h2 : c + d = 1)

theorem math_problem :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 + (d + 1 / d)^3 ≥ 40 := sorry

end math_problem_l599_59956


namespace bridge_length_l599_59934

noncomputable def speed_km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def distance_travelled (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_condition : train_length = 150) 
  (train_speed_condition : train_speed_kmph = 45) 
  (crossing_time_condition : crossing_time_s = 30) :
  (distance_travelled (speed_km_per_hr_to_m_per_s train_speed_kmph) crossing_time_s - train_length) = 225 :=
by 
  sorry

end bridge_length_l599_59934


namespace inverse_proposition_of_divisibility_by_5_l599_59985

theorem inverse_proposition_of_divisibility_by_5 (n : ℕ) :
  (n % 10 = 5 → n % 5 = 0) → (n % 5 = 0 → n % 10 = 5) :=
sorry

end inverse_proposition_of_divisibility_by_5_l599_59985


namespace students_prefer_windows_to_mac_l599_59904

-- Define the conditions
def total_students : ℕ := 210
def students_prefer_mac : ℕ := 60
def students_equally_prefer_both : ℕ := 20
def students_no_preference : ℕ := 90

-- The proof problem
theorem students_prefer_windows_to_mac :
  total_students - students_prefer_mac - students_equally_prefer_both - students_no_preference = 40 :=
by sorry

end students_prefer_windows_to_mac_l599_59904


namespace calculate_expression_l599_59910

theorem calculate_expression :
  500 * 1986 * 0.3972 * 100 = 20 * 1986^2 :=
by sorry

end calculate_expression_l599_59910


namespace bulbs_arrangement_l599_59957

theorem bulbs_arrangement :
  let blue_bulbs := 5
  let red_bulbs := 8
  let white_bulbs := 11
  let total_non_white_bulbs := blue_bulbs + red_bulbs
  let total_gaps := total_non_white_bulbs + 1
  (Nat.choose 13 5) * (Nat.choose total_gaps white_bulbs) = 468468 :=
by
  sorry

end bulbs_arrangement_l599_59957


namespace length_more_than_breadth_l599_59982

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end length_more_than_breadth_l599_59982


namespace value_of_m_l599_59919

theorem value_of_m (a a1 a2 a3 a4 a5 a6 m : ℝ) (x : ℝ)
  (h1 : (1 + m * x)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  (m = 1 ∨ m = -3) :=
sorry

end value_of_m_l599_59919


namespace min_positive_period_and_symmetry_axis_l599_59935

noncomputable def f (x : ℝ) := - (Real.sin (x + Real.pi / 6)) * (Real.sin (x - Real.pi / 3))

theorem min_positive_period_and_symmetry_axis :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∃ k : ℤ, ∀ x : ℝ, f x = f (x + 1 / 2 * k * Real.pi + Real.pi / 12)) := by
  sorry

end min_positive_period_and_symmetry_axis_l599_59935


namespace winnie_balloons_rem_l599_59914

theorem winnie_balloons_rem (r w g c : ℕ) (h_r : r = 17) (h_w : w = 33) (h_g : g = 65) (h_c : c = 83) :
  (r + w + g + c) % 8 = 6 := 
by 
  sorry

end winnie_balloons_rem_l599_59914


namespace max_area_of_rectangular_pen_l599_59980

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l599_59980


namespace evaluate_expression_l599_59989

def cyclical_i (z : ℂ) : Prop := z^4 = 1

theorem evaluate_expression (i : ℂ) (h : cyclical_i i) : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end evaluate_expression_l599_59989


namespace find_costs_compare_options_l599_59938

-- Definitions and theorems
def cost1 (x y : ℕ) : Prop := 2 * x + 4 * y = 350
def cost2 (x y : ℕ) : Prop := 6 * x + 3 * y = 420

def optionACost (m : ℕ) : ℕ := 70 * m + 35 * (80 - 2 * m)
def optionBCost (m : ℕ) : ℕ := (8 * (35 * m + 2800)) / 10

theorem find_costs (x y : ℕ) : 
  cost1 x y ∧ cost2 x y → (x = 35 ∧ y = 70) :=
by sorry

theorem compare_options (m : ℕ) (h : m < 41) : 
  if m < 20 then optionBCost m < optionACost m else 
  if m = 20 then optionBCost m = optionACost m 
  else optionBCost m > optionACost m :=
by sorry

end find_costs_compare_options_l599_59938
