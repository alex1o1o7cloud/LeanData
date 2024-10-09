import Mathlib

namespace laura_owes_amount_l311_31165

def principal : ℝ := 35
def rate : ℝ := 0.04
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount : ℝ := principal + interest

theorem laura_owes_amount :
  total_amount = 36.40 := by
  sorry

end laura_owes_amount_l311_31165


namespace jelly_bean_ratio_l311_31119

-- Define the number of jelly beans each person has
def napoleon_jelly_beans : ℕ := 17
def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4
def mikey_jelly_beans : ℕ := 19

-- Define the sum of jelly beans of Napoleon and Sedrich
def sum_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

-- Define the ratio of the sum of Napoleon and Sedrich's jelly beans to Mikey's jelly beans
def ratio : ℚ := sum_jelly_beans / mikey_jelly_beans

-- Prove that the ratio is 2
theorem jelly_bean_ratio : ratio = 2 := by
  -- We skip the proof steps since the focus here is on the correct statement
  sorry

end jelly_bean_ratio_l311_31119


namespace evaluate_f_at_5_l311_31106

def f (x : ℝ) := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 524

theorem evaluate_f_at_5 : f 5 = 2176 :=
by
  sorry

end evaluate_f_at_5_l311_31106


namespace dryer_weight_l311_31161

theorem dryer_weight 
(empty_truck_weight crates_soda_weight num_crates soda_weight_factor 
    fresh_produce_weight_factor num_dryers fully_loaded_truck_weight : ℕ) 

  (h1 : empty_truck_weight = 12000) 
  (h2 : crates_soda_weight = 50) 
  (h3 : num_crates = 20) 
  (h4 : soda_weight_factor = crates_soda_weight * num_crates) 
  (h5 : fresh_produce_weight_factor = 2 * soda_weight_factor) 
  (h6 : num_dryers = 3) 
  (h7 : fully_loaded_truck_weight = 24000) 

  : (fully_loaded_truck_weight - empty_truck_weight 
      - (soda_weight_factor + fresh_produce_weight_factor)) / num_dryers = 3000 := 
by sorry

end dryer_weight_l311_31161


namespace anderson_family_seating_l311_31185

def anderson_family_seating_arrangements : Prop :=
  ∃ (family : Fin 5 → String),
    (family 0 = "Mr. Anderson" ∨ family 0 = "Mrs. Anderson") ∧
    (∀ (i : Fin 5), i ≠ 0 → family i ≠ family 0) ∧
    family 1 ≠ family 0 ∧ (family 1 = "Mrs. Anderson" ∨ family 1 = "Child 1" ∨ family 1 = "Child 2") ∧
    family 2 = "Child 3" ∧
    (family 3 ≠ family 0 ∧ family 3 ≠ family 1 ∧ family 3 ≠ family 2) ∧
    (family 4 ≠ family 0 ∧ family 4 ≠ family 1 ∧ family 4 ≠ family 2 ∧ family 4 ≠ family 3) ∧
    (family 3 = "Child 1" ∨ family 3 = "Child 2") ∧
    (family 4 = "Child 1" ∨ family 4 = "Child 2") ∧
    family 3 ≠ family 4 → 
    (2 * 3 * 2 = 12)

theorem anderson_family_seating : anderson_family_seating_arrangements := 
  sorry

end anderson_family_seating_l311_31185


namespace students_who_wanted_fruit_l311_31158

theorem students_who_wanted_fruit (red_apples green_apples extra_apples ordered_apples served_apples students_wanted_fruit : ℕ)
    (h1 : red_apples = 43)
    (h2 : green_apples = 32)
    (h3 : extra_apples = 73)
    (h4 : ordered_apples = red_apples + green_apples)
    (h5 : served_apples = ordered_apples + extra_apples)
    (h6 : students_wanted_fruit = served_apples - ordered_apples) :
    students_wanted_fruit = 73 := 
by
    sorry

end students_who_wanted_fruit_l311_31158


namespace rectangle_dimensions_l311_31193

theorem rectangle_dimensions (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w * l) = 2 * (2 * w + w)) :
  w = 6 ∧ l = 12 := 
by sorry

end rectangle_dimensions_l311_31193


namespace find_a_l311_31122

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x + a

theorem find_a (a : ℝ) :
  (∃! x : ℝ, f x a = 0) → (a = -2 ∨ a = 2) :=
sorry

end find_a_l311_31122


namespace find_m_plus_M_l311_31121

-- Given conditions
def cond1 (x y z : ℝ) := x + y + z = 4
def cond2 (x y z : ℝ) := x^2 + y^2 + z^2 = 6

-- Proof statement: The sum of the smallest and largest possible values of x is 8/3
theorem find_m_plus_M :
  ∀ (x y z : ℝ), cond1 x y z → cond2 x y z → (min (x : ℝ) (max x y) + max (x : ℝ) (min x y) = 8 / 3) :=
by
  sorry

end find_m_plus_M_l311_31121


namespace trigonometric_identity_l311_31105

theorem trigonometric_identity : 
  Real.cos 6 * Real.cos 36 + Real.sin 6 * Real.cos 54 = Real.sqrt 3 / 2 :=
sorry

end trigonometric_identity_l311_31105


namespace minimum_nine_points_distance_l311_31198

theorem minimum_nine_points_distance (n : ℕ) : 
  (∀ (p : Fin n → ℝ × ℝ),
    (∀ i, ∃! (four_points : List (Fin n)), 
      List.length four_points = 4 ∧ (∀ j ∈ four_points, dist (p i) (p j) = 1)))
    ↔ n = 9 :=
by 
  sorry

end minimum_nine_points_distance_l311_31198


namespace not_necessarily_heavier_l311_31137

/--
In a zoo, there are 10 elephants. It is known that if any four elephants stand on the left pan and any three on the right pan, the left pan will weigh more. If five elephants stand on the left pan and four on the right pan, the left pan does not necessarily weigh more.
-/
theorem not_necessarily_heavier (E : Fin 10 → ℝ) (H : ∀ (L : Finset (Fin 10)) (R : Finset (Fin 10)), L.card = 4 → R.card = 3 → L ≠ R → L.sum E > R.sum E) :
  ∃ (L' R' : Finset (Fin 10)), L'.card = 5 ∧ R'.card = 4 ∧ L'.sum E ≤ R'.sum E :=
by
  sorry

end not_necessarily_heavier_l311_31137


namespace smallest_t_in_colored_grid_l311_31114

theorem smallest_t_in_colored_grid :
  ∃ (t : ℕ), (t > 0) ∧
  (∀ (coloring : Fin (100*100) → ℕ),
      (∀ (n : ℕ), (∃ (squares : Finset (Fin (100*100))), squares.card ≤ 104 ∧ ∀ x ∈ squares, coloring x = n)) →
      (∃ (rectangle : Finset (Fin (100*100))),
        (rectangle.card = t ∧ (t = 1 ∨ (t = 2 ∨ ∃ (l : ℕ), (l = 12 ∧ rectangle.card = l) ∧ (∃ (c : ℕ), (c = 3 ∧ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∃(s1 s2 s3 : Fin (100*100)), (s1 ∈ rectangle ∧ coloring s1 = a) ∧ (s2 ∈ rectangle ∧ coloring s2 = b) ∧ (s3 ∈ rectangle ∧ coloring s3 = c))))))))) :=
sorry

end smallest_t_in_colored_grid_l311_31114


namespace x_intercept_of_line_l311_31175

theorem x_intercept_of_line : ∃ x : ℚ, 3 * x + 5 * 0 = 20 ∧ (x, 0) = (20/3, 0) :=
by
  sorry

end x_intercept_of_line_l311_31175


namespace chord_eq_l311_31111

/-- 
If a chord of the ellipse x^2 / 36 + y^2 / 9 = 1 is bisected by the point (4,2),
then the equation of the line on which this chord lies is x + 2y - 8 = 0.
-/
theorem chord_eq {x y : ℝ} (H : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 / 36 + A.2 ^ 2 / 9 = 1) ∧ 
  (B.1 ^ 2 / 36 + B.2 ^ 2 / 9 = 1) ∧ 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4, 2)) :
  x + 2 * y = 8 :=
sorry

end chord_eq_l311_31111


namespace partI_solution_set_partII_range_of_m_l311_31170

def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

theorem partI_solution_set (x : ℝ) :
  ∀ (x : ℝ), f x 5 ≤ 12 ↔ (-13 / 2 ≤ x ∧ x ≤ 11 / 2) :=
by
  sorry

theorem partII_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end partI_solution_set_partII_range_of_m_l311_31170


namespace time_to_ascend_non_working_escalator_l311_31159

-- Define the variables as given in the conditions
def V := 1 / 60 -- Speed of the moving escalator in units per minute
def U := (1 / 24) - (1 / 60) -- Speed of Gavrila running relative to the escalator

-- Theorem stating that the time to ascend a non-working escalator is 40 seconds
theorem time_to_ascend_non_working_escalator : 
  (1 : ℚ) = U * (40 / 60) := 
by sorry

end time_to_ascend_non_working_escalator_l311_31159


namespace percent_decrease_computer_price_l311_31166

theorem percent_decrease_computer_price (price_1990 price_2010 : ℝ) (h1 : price_1990 = 1200) (h2 : price_2010 = 600) :
  ((price_1990 - price_2010) / price_1990) * 100 = 50 := 
  sorry

end percent_decrease_computer_price_l311_31166


namespace Randy_bats_l311_31148

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end Randy_bats_l311_31148


namespace count_integers_l311_31189

theorem count_integers (n : ℤ) (h : -11 ≤ n ∧ n ≤ 11) : ∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x ∈ s, (x - 1) * (x + 3) * (x + 7) < 0 :=
by
  sorry

end count_integers_l311_31189


namespace Seth_gave_to_his_mother_l311_31173

variable (x : ℕ)

-- Define the conditions as per the problem statement
def initial_boxes := 9
def remaining_boxes_after_giving_to_mother := initial_boxes - x
def remaining_boxes_after_giving_half := remaining_boxes_after_giving_to_mother / 2

-- Specify the final condition
def final_boxes := 4

-- Form the main theorem
theorem Seth_gave_to_his_mother :
  final_boxes = remaining_boxes_after_giving_to_mother / 2 →
  initial_boxes - x = 8 :=
by sorry

end Seth_gave_to_his_mother_l311_31173


namespace bottle_caps_per_friend_l311_31123

-- The context where Catherine has 18 bottle caps
def bottle_caps : Nat := 18

-- Catherine distributes these bottle caps among 6 friends
def number_of_friends : Nat := 6

-- We need to prove that each friend gets 3 bottle caps
theorem bottle_caps_per_friend : bottle_caps / number_of_friends = 3 :=
by sorry

end bottle_caps_per_friend_l311_31123


namespace negation_of_universal_l311_31127

theorem negation_of_universal :
  (¬ (∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1)) ↔ 
  (∃ k : ℝ, ¬ ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1) :=
by
  sorry

end negation_of_universal_l311_31127


namespace pencil_cost_l311_31154

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (h1 : total_money = 50) (h2 : num_pencils = 10) :
    (total_money / num_pencils) = 5 :=
by
  sorry

end pencil_cost_l311_31154


namespace city_population_l311_31199

theorem city_population (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 :=
by
  sorry

end city_population_l311_31199


namespace f_0_eq_0_l311_31116

-- Define a function f with the given condition
def f (x : ℤ) : ℤ := if x = 0 then 0
                     else (x-1)^2 + 2*(x-1) + 1

-- State the theorem
theorem f_0_eq_0 : f 0 = 0 :=
by sorry

end f_0_eq_0_l311_31116


namespace history_books_count_l311_31174

-- Definitions based on conditions
def total_books : Nat := 100
def geography_books : Nat := 25
def math_books : Nat := 43

-- Problem statement: proving the number of history books
theorem history_books_count : total_books - geography_books - math_books = 32 := by
  sorry

end history_books_count_l311_31174


namespace problem1_problem2_l311_31195

-- Given conditions
variables (x y : ℝ)

-- Problem 1: Prove that ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = -xy
theorem problem1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = - (x * y) :=
sorry

-- Problem 2: Prove that (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2
theorem problem2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
sorry

end problem1_problem2_l311_31195


namespace chord_eq_line_l311_31190

theorem chord_eq_line (x y : ℝ)
  (h_ellipse : (x^2) / 16 + (y^2) / 4 = 1)
  (h_midpoint : ∃ x1 y1 x2 y2 : ℝ, 
    ((x1^2) / 16 + (y1^2) / 4 = 1) ∧ 
    ((x2^2) / 16 + (y2^2) / 4 = 1) ∧ 
    (x1 + x2) / 2 = 2 ∧ 
    (y1 + y2) / 2 = 1) :
  x + 2 * y - 4 = 0 :=
sorry

end chord_eq_line_l311_31190


namespace geometric_sum_n_is_4_l311_31186

theorem geometric_sum_n_is_4 
  (a r : ℚ) (n : ℕ) (S_n : ℚ) 
  (h1 : a = 1) 
  (h2 : r = 1 / 4) 
  (h3 : S_n = (a * (1 - r^n)) / (1 - r)) 
  (h4 : S_n = 85 / 64) : 
  n = 4 := 
sorry

end geometric_sum_n_is_4_l311_31186


namespace rectangle_fitting_condition_l311_31145

variables {a b c d : ℝ}

theorem rectangle_fitting_condition
  (h1: a < c ∧ c ≤ d ∧ d < b)
  (h2: a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 :=
sorry

end rectangle_fitting_condition_l311_31145


namespace difference_in_height_l311_31149

-- Define the heights of the sandcastles
def h_J : ℚ := 3.6666666666666665
def h_S : ℚ := 2.3333333333333335

-- State the theorem
theorem difference_in_height :
  h_J - h_S = 1.333333333333333 := by
  sorry

end difference_in_height_l311_31149


namespace probability_of_same_color_is_34_over_105_l311_31150

-- Define the number of each color of plates
def num_red_plates : ℕ := 7
def num_blue_plates : ℕ := 5
def num_yellow_plates : ℕ := 3

-- Define the total number of plates
def total_plates : ℕ := num_red_plates + num_blue_plates + num_yellow_plates

-- Define the total number of ways to choose 2 plates from the total plates
def total_ways_to_choose_2_plates : ℕ := Nat.choose total_plates 2

-- Define the number of ways to choose 2 red plates, 2 blue plates, and 2 yellow plates
def ways_to_choose_2_red_plates : ℕ := Nat.choose num_red_plates 2
def ways_to_choose_2_blue_plates : ℕ := Nat.choose num_blue_plates 2
def ways_to_choose_2_yellow_plates : ℕ := Nat.choose num_yellow_plates 2

-- Define the total number of favorable outcomes (same color plates)
def favorable_outcomes : ℕ :=
  ways_to_choose_2_red_plates + ways_to_choose_2_blue_plates + ways_to_choose_2_yellow_plates

-- Prove that the probability is 34/105
theorem probability_of_same_color_is_34_over_105 :
  (favorable_outcomes : ℚ) / (total_ways_to_choose_2_plates : ℚ) = 34 / 105 := by
  sorry

end probability_of_same_color_is_34_over_105_l311_31150


namespace cubic_roots_sum_cube_l311_31140

theorem cubic_roots_sum_cube (a b c : ℂ) (h : ∀x : ℂ, (x=a ∨ x=b ∨ x=c) → (x^3 - 2*x^2 + 3*x - 4 = 0)) : a^3 + b^3 + c^3 = 2 :=
sorry

end cubic_roots_sum_cube_l311_31140


namespace price_of_each_cake_is_correct_l311_31187

-- Define the conditions
def total_flour : ℕ := 6
def flour_for_cakes : ℕ := 4
def flour_per_cake : ℚ := 0.5
def remaining_flour := total_flour - flour_for_cakes
def flour_per_cupcake : ℚ := 1 / 5
def total_earnings : ℚ := 30
def cupcake_price : ℚ := 1

-- Number of cakes and cupcakes
def number_of_cakes := flour_for_cakes / flour_per_cake
def number_of_cupcakes := remaining_flour / flour_per_cupcake

-- Earnings from cupcakes
def earnings_from_cupcakes := number_of_cupcakes * cupcake_price

-- Earnings from cakes
def earnings_from_cakes := total_earnings - earnings_from_cupcakes

-- Price per cake
def price_per_cake := earnings_from_cakes / number_of_cakes

-- Final statement to prove
theorem price_of_each_cake_is_correct : price_per_cake = 2.50 := by
  sorry

end price_of_each_cake_is_correct_l311_31187


namespace chord_length_of_tangent_circle_l311_31101

theorem chord_length_of_tangent_circle
  (area_of_ring : ℝ)
  (diameter_large_circle : ℝ)
  (h1 : area_of_ring = (50 / 3) * Real.pi)
  (h2 : diameter_large_circle = 10) :
  ∃ (length_of_chord : ℝ), length_of_chord = (10 * Real.sqrt 6) / 3 := by
  sorry

end chord_length_of_tangent_circle_l311_31101


namespace slope_of_line_l311_31139

theorem slope_of_line : 
  (∀ x y : ℝ, (y = (1/2) * x + 1) → ∃ m : ℝ, m = 1/2) :=
sorry

end slope_of_line_l311_31139


namespace seq_an_identity_l311_31152

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end seq_an_identity_l311_31152


namespace range_of_a_l311_31177

theorem range_of_a (a : ℝ) (h : Real.sqrt ((2 * a - 1)^2) = 1 - 2 * a) : a ≤ 1 / 2 :=
sorry

end range_of_a_l311_31177


namespace emmalyn_earnings_l311_31142

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l311_31142


namespace total_time_for_seven_flights_l311_31115

theorem total_time_for_seven_flights :
  let a := 15
  let d := 8
  let n := 7
  let l := a + (n - 1) * d
  let S_n := n * (a + l) / 2
  S_n = 273 :=
by
  sorry

end total_time_for_seven_flights_l311_31115


namespace dismissed_cases_l311_31169

theorem dismissed_cases (total_cases : Int) (X : Int)
  (total_cases_eq : total_cases = 17)
  (remaining_cases_eq : X = (2 * X / 3) + 1 + 4) :
  total_cases - X = 2 :=
by
  -- Placeholder for the proof
  sorry

end dismissed_cases_l311_31169


namespace most_appropriate_survey_is_D_l311_31109

-- Define the various scenarios as Lean definitions
def survey_A := "Testing whether a certain brand of fresh milk meets food hygiene standards, using a census method."
def survey_B := "Security check before taking the subway, using a sampling survey method."
def survey_C := "Understanding the sleep time of middle school students in Jiangsu Province, using a census method."
def survey_D := "Understanding the way Nanjing residents commemorate the Qingming Festival, using a sampling survey method."

-- Define the type for specifying which survey method is the most appropriate
def appropriate_survey (survey : String) : Prop := 
  survey = survey_D

-- The theorem statement proving that the most appropriate survey is D
theorem most_appropriate_survey_is_D : appropriate_survey survey_D :=
by sorry

end most_appropriate_survey_is_D_l311_31109


namespace flowers_brought_at_dawn_l311_31132

theorem flowers_brought_at_dawn (F : ℕ) 
  (h1 : (3 / 5) * F = 180)
  (h2 :  (2 / 5) * F + (F - (3 / 5) * F) = 180) : 
  F = 300 := 
by
  sorry

end flowers_brought_at_dawn_l311_31132


namespace flea_returns_to_0_l311_31179

noncomputable def flea_return_probability (p : ℝ) : ℝ :=
if p = 1 then 0 else 1

theorem flea_returns_to_0 (p : ℝ) : 
  flea_return_probability p = (if p = 1 then 0 else 1) :=
by
  sorry

end flea_returns_to_0_l311_31179


namespace expressions_cannot_all_exceed_one_fourth_l311_31134

theorem expressions_cannot_all_exceed_one_fourth (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := 
by
  sorry

end expressions_cannot_all_exceed_one_fourth_l311_31134


namespace goshawk_eurasian_reserve_l311_31182

theorem goshawk_eurasian_reserve (B : ℝ)
  (h1 : 0.30 * B + 0.28 * B + K * 0.28 * B = 0.65 * B)
  : K = 0.25 :=
by sorry

end goshawk_eurasian_reserve_l311_31182


namespace last_four_digits_of_5_pow_2013_l311_31128

theorem last_four_digits_of_5_pow_2013 : (5 ^ 2013) % 10000 = 3125 :=
by
  sorry

end last_four_digits_of_5_pow_2013_l311_31128


namespace not_divisible_l311_31178

-- Defining the necessary conditions
variable (m : ℕ)

theorem not_divisible (m : ℕ) : ¬ (1000^m - 1 ∣ 1978^m - 1) :=
sorry

end not_divisible_l311_31178


namespace find_certain_number_l311_31131

theorem find_certain_number : ∃ x : ℕ, (((x - 50) / 4) * 3 + 28 = 73) → x = 110 :=
by
  sorry

end find_certain_number_l311_31131


namespace incorrect_proposition_b_l311_31188

axiom plane (α β : Type) : Prop
axiom line (m n : Type) : Prop
axiom parallel (a b : Type) : Prop
axiom perpendicular (a b : Type) : Prop
axiom intersection (α β : Type) (n : Type) : Prop
axiom contained (a b : Type) : Prop

theorem incorrect_proposition_b (α β m n : Type)
  (hαβ_plane : plane α β)
  (hmn_line : line m n)
  (h_parallel_m_α : parallel m α)
  (h_intersection : intersection α β n) :
  ¬ parallel m n :=
sorry

end incorrect_proposition_b_l311_31188


namespace total_gallons_needed_l311_31172

def gas_can_capacity : ℝ := 5.0
def number_of_cans : ℝ := 4.0
def total_gallons_of_gas : ℝ := gas_can_capacity * number_of_cans

theorem total_gallons_needed : total_gallons_of_gas = 20.0 := by
  -- proof goes here
  sorry

end total_gallons_needed_l311_31172


namespace tourist_groups_meet_l311_31156

theorem tourist_groups_meet (x y : ℝ) (h1 : 4.5 * x + 2.5 * y = 30) (h2 : 3 * x + 5 * y = 30) : 
  x = 5 ∧ y = 3 := 
sorry

end tourist_groups_meet_l311_31156


namespace mail_cars_in_train_l311_31184

theorem mail_cars_in_train (n : ℕ) (hn : n % 2 = 0) (hfront : 1 ≤ n ∧ n ≤ 20)
  (hclose : ∀ i, 1 ≤ i ∧ i < n → (∃ j, i < j ∧ j ≤ 20))
  (hlast : 4 * n ≤ 20)
  (hconn : ∀ k, (k = 4 ∨ k = 5 ∨ k = 15 ∨ k = 16) → 
                  (∃ j, j = k + 1 ∨ j = k - 1)) :
  ∃ (i : ℕ) (j : ℕ), i = 4 ∧ j = 16 :=
by
  sorry

end mail_cars_in_train_l311_31184


namespace gcf_75_100_l311_31141

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l311_31141


namespace count_three_digit_congruent_to_5_mod_7_l311_31133

theorem count_three_digit_congruent_to_5_mod_7 : 
  (100 ≤ 7 * k + 5 ∧ 7 * k + 5 ≤ 999) → ∃ n : ℕ, n = 129 := sorry

end count_three_digit_congruent_to_5_mod_7_l311_31133


namespace cubic_yard_to_cubic_meter_and_liters_l311_31110

theorem cubic_yard_to_cubic_meter_and_liters :
  (1 : ℝ) * (0.9144 : ℝ)^3 = 0.764554 ∧ 0.764554 * 1000 = 764.554 :=
by
  sorry

end cubic_yard_to_cubic_meter_and_liters_l311_31110


namespace arithmetic_sequence_sum_l311_31191

variable {a_n : ℕ → ℕ} -- the arithmetic sequence

-- Define condition
def condition (a : ℕ → ℕ) : Prop :=
  a 1 + a 5 + a 9 = 18

-- The sum of the first n terms of arithmetic sequence is S_n
def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

-- The goal is to prove that S 9 = 54
theorem arithmetic_sequence_sum (h : condition a_n) : S 9 a_n = 54 :=
sorry

end arithmetic_sequence_sum_l311_31191


namespace correct_proposition_l311_31129

theorem correct_proposition : 
  (¬ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 2 * x_0) ↔ (∀ x : ℝ, x^2 + 1 > 2 * x) := 
sorry

end correct_proposition_l311_31129


namespace greatest_QPN_value_l311_31138

theorem greatest_QPN_value (N : ℕ) (Q P : ℕ) (QPN : ℕ) :
  (NN : ℕ) =
  10 * N + N ∧
  QPN = 100 * Q + 10 * P + N ∧
  N < 10 ∧ N ≥ 1 ∧
  NN * N = QPN ∧
  NN >= 10 ∧ NN < 100  -- Ensuring NN is a two-digit number
  → QPN <= 396 := sorry

end greatest_QPN_value_l311_31138


namespace arctg_inequality_l311_31153

theorem arctg_inequality (a b : ℝ) :
    |Real.arctan a - Real.arctan b| ≤ |b - a| := 
sorry

end arctg_inequality_l311_31153


namespace problem_a_problem_b_l311_31180

theorem problem_a (p : ℕ) (hp : Nat.Prime p) : 
  (∃ x : ℕ, (7^(p-1) - 1) = p * x^2) ↔ p = 3 := 
by
  sorry

theorem problem_b (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ x : ℕ, (11^(p-1) - 1) = p * x^2 := 
by
  sorry

end problem_a_problem_b_l311_31180


namespace grain_remaining_l311_31120

def originalGrain : ℕ := 50870
def spilledGrain : ℕ := 49952
def remainingGrain : ℕ := 918

theorem grain_remaining : originalGrain - spilledGrain = remainingGrain := by
  -- calculations are omitted in the theorem statement
  sorry

end grain_remaining_l311_31120


namespace sixteen_k_plus_eight_not_perfect_square_l311_31196

theorem sixteen_k_plus_eight_not_perfect_square (k : ℕ) (hk : 0 < k) : ¬ ∃ m : ℕ, (16 * k + 8) = m * m := sorry

end sixteen_k_plus_eight_not_perfect_square_l311_31196


namespace perfect_square_condition_l311_31113

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l311_31113


namespace course_count_l311_31108

theorem course_count (n1 n2 : ℕ) (sum_x1 sum_x2 : ℕ) :
  (n1 = 6) →
  (sum_x1 = n1 * 100) →
  (sum_x2 = n2 * 50) →
  ((sum_x1 + sum_x2) / (n1 + n2) = 77) →
  n2 = 5 :=
by
  intros h1 h2 h3 h4
  sorry

end course_count_l311_31108


namespace calculate_gross_profit_l311_31151

theorem calculate_gross_profit (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 81)
    (h2 : gross_profit = 1.70 * cost)
    (h3 : sales_price = cost + gross_profit) : gross_profit = 51 :=
by
  sorry

end calculate_gross_profit_l311_31151


namespace no_valid_two_digit_factors_l311_31167

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Main theorem to show: there are no valid two-digit factorizations of 1976
theorem no_valid_two_digit_factors : 
  ∃ (factors : ℕ → ℕ → Prop), (∀ (a b : ℕ), factors a b → (a * b = 1976) → (is_two_digit a) → (is_two_digit b)) → 
  ∃ (count : ℕ), count = 0 := 
sorry

end no_valid_two_digit_factors_l311_31167


namespace fraction_exponentiation_l311_31157

theorem fraction_exponentiation :
  (⟨1/3⟩ : ℝ) ^ 5 = (⟨1/243⟩ : ℝ) :=
by
  sorry

end fraction_exponentiation_l311_31157


namespace aku_invited_friends_l311_31126

def total_cookies (packages : ℕ) (cookies_per_package : ℕ) := packages * cookies_per_package

def total_children (total_cookies : ℕ) (cookies_per_child : ℕ) := total_cookies / cookies_per_child

def invited_friends (total_children : ℕ) := total_children - 1

theorem aku_invited_friends (packages cookies_per_package cookies_per_child : ℕ) (h1 : packages = 3) (h2 : cookies_per_package = 25) (h3 : cookies_per_child = 15) :
  invited_friends (total_children (total_cookies packages cookies_per_package) cookies_per_child) = 4 :=
by
  sorry

end aku_invited_friends_l311_31126


namespace product_of_roots_eq_25_l311_31192

theorem product_of_roots_eq_25 (t : ℝ) (h : t^2 - 10 * t + 25 = 0) : t * t = 25 :=
sorry

end product_of_roots_eq_25_l311_31192


namespace eq_x4_inv_x4_l311_31181

theorem eq_x4_inv_x4 (x : ℝ) (h : x^2 + (1 / x^2) = 2) : 
  x^4 + (1 / x^4) = 2 := 
by 
  sorry

end eq_x4_inv_x4_l311_31181


namespace most_compliant_expression_l311_31160

-- Define the expressions as algebraic terms.
def OptionA : String := "1(1/2)a"
def OptionB : String := "b/a"
def OptionC : String := "3a-1 个"
def OptionD : String := "a * 3"

-- Define a property that represents compliance with standard algebraic notation.
def is_compliant (expr : String) : Prop :=
  expr = OptionB

-- The theorem to prove.
theorem most_compliant_expression :
  is_compliant OptionB :=
by
  sorry

end most_compliant_expression_l311_31160


namespace minimize_maximum_absolute_value_expression_l311_31183

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l311_31183


namespace parking_garage_savings_l311_31194

theorem parking_garage_savings :
  let weekly_cost := 10
  let monthly_cost := 35
  let weeks_per_year := 52
  let months_per_year := 12
  let annual_weekly_cost := weekly_cost * weeks_per_year
  let annual_monthly_cost := monthly_cost * months_per_year
  let annual_savings := annual_weekly_cost - annual_monthly_cost
  annual_savings = 100 := 
by
  sorry

end parking_garage_savings_l311_31194


namespace sufficient_but_not_necessary_condition_l311_31155

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≤ 0) : 
  a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_l311_31155


namespace percentage_markup_l311_31100

theorem percentage_markup (sell_price : ℝ) (cost_price : ℝ)
  (h_sell : sell_price = 8450) (h_cost : cost_price = 6500) : 
  (sell_price - cost_price) / cost_price * 100 = 30 :=
by
  sorry

end percentage_markup_l311_31100


namespace trader_sells_cloth_l311_31107

variable (x : ℝ) (SP_total : ℝ := 6900) (profit_per_meter : ℝ := 20) (CP_per_meter : ℝ := 66.25)

theorem trader_sells_cloth : SP_total = x * (CP_per_meter + profit_per_meter) → x = 80 :=
by
  intro h
  -- Placeholder for actual proof
  sorry

end trader_sells_cloth_l311_31107


namespace coffee_table_price_correct_l311_31162

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end coffee_table_price_correct_l311_31162


namespace sums_have_same_remainder_l311_31125

theorem sums_have_same_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) : 
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i.val) % (2 * n) = (a j + j.val) % (2 * n)) := 
sorry

end sums_have_same_remainder_l311_31125


namespace muffin_to_banana_ratio_l311_31147

variables (m b : ℝ) -- initial cost of a muffin and a banana

-- John's total cost for muffins and bananas
def johns_cost (m b : ℝ) : ℝ :=
  3 * m + 4 * b

-- Martha's total cost for muffins and bananas based on increased prices
def marthas_cost_increased (m b : ℝ) : ℝ :=
  5 * (1.2 * m) + 12 * (1.5 * b)

-- John's total cost times three
def marthas_cost_original_times_three (m b : ℝ) : ℝ :=
  3 * (johns_cost m b)

-- The theorem to prove
theorem muffin_to_banana_ratio
  (h3m4b_eq : johns_cost m b * 3 = marthas_cost_increased m b)
  (hm_eq_2b : m = 2 * b) :
  (1.2 * m) / (1.5 * b) = 4 / 5 := by
  sorry

end muffin_to_banana_ratio_l311_31147


namespace bananas_in_collection_l311_31130

theorem bananas_in_collection
  (groups : ℕ)
  (bananas_per_group : ℕ)
  (h1 : groups = 11)
  (h2 : bananas_per_group = 37) :
  (groups * bananas_per_group) = 407 :=
by sorry

end bananas_in_collection_l311_31130


namespace scout_troop_net_profit_l311_31118

theorem scout_troop_net_profit :
  ∃ (cost_per_bar selling_price_per_bar : ℝ),
    cost_per_bar = 1 / 3 ∧
    selling_price_per_bar = 0.6 ∧
    (1500 * selling_price_per_bar - (1500 * cost_per_bar + 50) = 350) :=
by {
  sorry
}

end scout_troop_net_profit_l311_31118


namespace percent_area_square_in_rectangle_l311_31197

theorem percent_area_square_in_rectangle
  (s : ℝ) 
  (w : ℝ) 
  (l : ℝ)
  (h1 : w = 3 * s) 
  (h2 : l = (9 / 2) * s) 
  : (s^2 / (l * w)) * 100 = 7.41 :=
by
  sorry

end percent_area_square_in_rectangle_l311_31197


namespace calculate_expression_l311_31163

theorem calculate_expression : (3.75 - 1.267 + 0.48 = 2.963) :=
by
  sorry

end calculate_expression_l311_31163


namespace number_of_integer_segments_l311_31117

theorem number_of_integer_segments (DE EF : ℝ) (H1 : DE = 24) (H2 : EF = 25) : 
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_integer_segments_l311_31117


namespace correct_statement_l311_31112

theorem correct_statement (x : ℝ) : 
  (∃ y : ℝ, y ≠ 0 ∧ y * x = 1 → x = 1 ∨ x = -1 ∨ x = 0) → false ∧
  (∃ y : ℝ, -y = y → y = 0 ∨ y = 1) → false ∧
  (abs x = x → x ≥ 0) → (x ^ 2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end correct_statement_l311_31112


namespace simplify_neg_x_mul_3_minus_x_l311_31102

theorem simplify_neg_x_mul_3_minus_x (x : ℝ) : -x * (3 - x) = -3 * x + x^2 :=
by
  sorry

end simplify_neg_x_mul_3_minus_x_l311_31102


namespace problem_lean_l311_31171

variable (α : ℝ)

-- Given condition
axiom given_cond : (1 + Real.sin α) * (1 - Real.cos α) = 1

-- Proof to be proven
theorem problem_lean : (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end problem_lean_l311_31171


namespace max_value_neg_domain_l311_31136

theorem max_value_neg_domain (x : ℝ) (h : x < 0) : 
  ∃ y, y = 2 * x + 2 / x ∧ y ≤ -4 :=
sorry

end max_value_neg_domain_l311_31136


namespace magdalena_fraction_picked_l311_31124

noncomputable def fraction_picked_first_day
  (produced_apples: ℕ)
  (remaining_apples: ℕ)
  (fraction_picked: ℚ) : Prop :=
  ∃ (f : ℚ),
  produced_apples = 200 ∧
  remaining_apples = 20 ∧
  (f = fraction_picked) ∧
  (200 * f + 2 * 200 * f + (200 * f + 20)) = 200 - remaining_apples ∧
  fraction_picked = 1 / 5

theorem magdalena_fraction_picked :
  fraction_picked_first_day 200 20 (1 / 5) :=
sorry

end magdalena_fraction_picked_l311_31124


namespace find_two_digit_number_l311_31146

theorem find_two_digit_number (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : n % 2 = 0)
  (h3 : (n + 1) % 3 = 0)
  (h4 : (n + 2) % 4 = 0)
  (h5 : (n + 3) % 5 = 0) : n = 62 :=
by
  sorry

end find_two_digit_number_l311_31146


namespace maximum_possible_value_of_expression_l311_31164

theorem maximum_possible_value_of_expression :
  ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) ∧
  (b = 0 ∨ b = 1 ∨ b = 3 ∨ b = 4) ∧
  (c = 0 ∨ c = 1 ∨ c = 3 ∨ c = 4) ∧
  (d = 0 ∨ d = 1 ∨ d = 3 ∨ d = 4) ∧
  ¬ (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) →
  (c * a^b + d ≤ 196) :=
by sorry

end maximum_possible_value_of_expression_l311_31164


namespace largest_prime_factor_l311_31144

theorem largest_prime_factor (a b c d : ℕ) (ha : a = 20) (hb : b = 15) (hc : c = 10) (hd : d = 5) :
  ∃ p, Nat.Prime p ∧ p = 103 ∧ ∀ q, Nat.Prime q ∧ q ∣ (a^3 + b^4 - c^5 + d^6) → q ≤ p :=
by
  sorry

end largest_prime_factor_l311_31144


namespace range_of_a_for_inequality_l311_31104

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end range_of_a_for_inequality_l311_31104


namespace fraction_difference_l311_31176

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l311_31176


namespace max_value_of_f_prime_div_f_l311_31168

def f (x : ℝ) : ℝ := sorry

theorem max_value_of_f_prime_div_f (f : ℝ → ℝ) (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x) (h2 : f 0 = 1) :
  ∀ x > 0, (deriv f x / f x) ≤ 2 :=
sorry

end max_value_of_f_prime_div_f_l311_31168


namespace find_a8_l311_31143

variable (a : ℕ → ℤ)

axiom h1 : ∀ n : ℕ, 2 * a n + a (n + 1) = 0
axiom h2 : a 3 = -2

theorem find_a8 : a 8 = 64 := by
  sorry

end find_a8_l311_31143


namespace base_8_to_decimal_77_eq_63_l311_31135

-- Define the problem in Lean 4
theorem base_8_to_decimal_77_eq_63 (k a1 a2 : ℕ) (h_k : k = 8) (h_a1 : a1 = 7) (h_a2 : a2 = 7) :
    a2 * k^1 + a1 * k^0 = 63 := 
by
  -- Placeholder for proof
  sorry

end base_8_to_decimal_77_eq_63_l311_31135


namespace right_triangle_sides_unique_l311_31103

theorem right_triangle_sides_unique (a b c : ℕ) 
  (relatively_prime : Int.gcd (Int.gcd a b) c = 1) 
  (right_triangle : a ^ 2 + b ^ 2 = c ^ 2) 
  (increased_right_triangle : (a + 100) ^ 2 + (b + 100) ^ 2 = (c + 140) ^ 2) : 
  (a = 56 ∧ b = 33 ∧ c = 65) :=
by
  sorry 

end right_triangle_sides_unique_l311_31103
