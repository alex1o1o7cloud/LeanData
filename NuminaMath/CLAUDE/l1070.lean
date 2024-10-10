import Mathlib

namespace midpoint_after_translation_l1070_107078

/-- Given a triangle DJH with vertices D(2, 3), J(3, 7), and H(7, 3),
    prove that the midpoint of D'H' after translating the triangle
    3 units right and 1 unit down is (7.5, 2). -/
theorem midpoint_after_translation :
  let D : ℝ × ℝ := (2, 3)
  let J : ℝ × ℝ := (3, 7)
  let H : ℝ × ℝ := (7, 3)
  let translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 - 1)
  let D' := translate D
  let H' := translate H
  let midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  midpoint D' H' = (7.5, 2) :=
by sorry

end midpoint_after_translation_l1070_107078


namespace largest_angle_in_triangle_l1070_107075

theorem largest_angle_in_triangle (X Y Z : Real) (h_scalene : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
  (h_angleY : Y = 25) (h_angleZ : Z = 100) : 
  max X (max Y Z) = 100 :=
sorry

end largest_angle_in_triangle_l1070_107075


namespace triangle_mapping_l1070_107051

theorem triangle_mapping :
  ∃ (f : ℂ → ℂ), 
    (∀ z w, f z = w ↔ w = (1 + Complex.I) * (1 - z)) ∧
    f 0 = 1 + Complex.I ∧
    f 1 = 0 ∧
    f Complex.I = 2 :=
by sorry

end triangle_mapping_l1070_107051


namespace ufo_convention_attendees_l1070_107088

theorem ufo_convention_attendees 
  (total_attendees : ℕ) 
  (total_presenters : ℕ) 
  (male_presenters female_presenters : ℕ) 
  (male_general female_general : ℕ) :
  total_attendees = 1000 →
  total_presenters = 420 →
  male_presenters = female_presenters + 20 →
  female_general = male_general + 56 →
  total_attendees = total_presenters + male_general + female_general →
  male_general = 262 := by
sorry

end ufo_convention_attendees_l1070_107088


namespace consecutive_pair_divisible_by_five_l1070_107020

theorem consecutive_pair_divisible_by_five (a b : ℕ) : 
  a < 1500 → 
  b < 1500 → 
  b = a + 1 → 
  (a + b) % 5 = 0 → 
  a = 57 → 
  b = 58 := by
sorry

end consecutive_pair_divisible_by_five_l1070_107020


namespace theme_parks_sum_l1070_107072

theorem theme_parks_sum (jamestown venice marina_del_ray : ℕ) : 
  jamestown = 20 →
  venice = jamestown + 25 →
  marina_del_ray = jamestown + 50 →
  jamestown + venice + marina_del_ray = 135 := by
  sorry

end theme_parks_sum_l1070_107072


namespace james_flowers_l1070_107073

/-- The number of flowers planted by James and his friends --/
def flower_planting 
  (james friend_a friend_b friend_c friend_d friend_e friend_f friend_g : ℝ) : Prop :=
  james = friend_a * 1.2
  ∧ friend_a = friend_b * 1.15
  ∧ friend_b = friend_c * 0.7
  ∧ friend_c = friend_d * 1.1
  ∧ friend_d = friend_e * 1.25
  ∧ friend_e = friend_f
  ∧ friend_g = friend_f * 0.7
  ∧ friend_b = 12

/-- The theorem stating James plants 16.56 flowers per day --/
theorem james_flowers 
  (james friend_a friend_b friend_c friend_d friend_e friend_f friend_g : ℝ) 
  (h : flower_planting james friend_a friend_b friend_c friend_d friend_e friend_f friend_g) : 
  james = 16.56 := by
  sorry

end james_flowers_l1070_107073


namespace wave_number_probability_l1070_107092

/-- A permutation of the digits 1,2,3,4,5 --/
def Permutation := Fin 5 → Fin 5

/-- A permutation is valid if it's bijective --/
def is_valid_permutation (p : Permutation) : Prop :=
  Function.Bijective p

/-- A permutation represents a wave number if it satisfies the wave pattern --/
def is_wave_number (p : Permutation) : Prop :=
  p 0 < p 1 ∧ p 1 > p 2 ∧ p 2 < p 3 ∧ p 3 > p 4

/-- The total number of valid permutations --/
def total_permutations : ℕ := 120

/-- The number of wave numbers --/
def wave_numbers : ℕ := 16

/-- The main theorem: probability of selecting a wave number --/
theorem wave_number_probability :
  (wave_numbers : ℚ) / total_permutations = 2 / 15 := by sorry

end wave_number_probability_l1070_107092


namespace simplified_expression_l1070_107067

theorem simplified_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 := by
  sorry

end simplified_expression_l1070_107067


namespace solution_set_equality_l1070_107086

def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ n = (p : ℤ) ^ (k : ℕ)

theorem solution_set_equality : 
  {x : ℤ | ∃ (y : ℤ), y > 0 ∧ is_prime_power y ∧ f x = y} = {-3, 2, 5} := by sorry

end solution_set_equality_l1070_107086


namespace cubic_factorization_l1070_107096

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end cubic_factorization_l1070_107096


namespace only_set_C_forms_triangle_l1070_107022

/-- A function that checks if three numbers can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of lengths given in the problem --/
def set_A : List ℝ := [3, 4, 8]
def set_B : List ℝ := [8, 7, 15]
def set_C : List ℝ := [13, 12, 20]
def set_D : List ℝ := [5, 5, 11]

/-- Theorem stating that only set C can form a triangle --/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  can_form_triangle set_C[0] set_C[1] set_C[2] ∧
  ¬(can_form_triangle set_D[0] set_D[1] set_D[2]) :=
by sorry

end only_set_C_forms_triangle_l1070_107022


namespace solution_to_diophantine_equation_l1070_107093

theorem solution_to_diophantine_equation :
  ∀ x y z : ℕ+,
    x ≤ y ∧ y ≤ z →
    5 * (x * y + y * z + z * x) = 4 * x * y * z →
    ((x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20)) :=
by sorry

end solution_to_diophantine_equation_l1070_107093


namespace valid_numbers_count_l1070_107087

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of 6-digit positive integers with digits from 1 to 6 in increasing order --/
def total_increasing_numbers : ℕ := stars_and_bars 6 6

/-- The number of 5-digit positive integers with digits from 1 to 5 in increasing order --/
def numbers_starting_with_6 : ℕ := stars_and_bars 5 5

/-- The number of 6-digit positive integers with digits from 1 to 6 in increasing order, not starting with 6 --/
def valid_numbers : ℕ := total_increasing_numbers - numbers_starting_with_6

theorem valid_numbers_count : valid_numbers = 336 := by
  sorry

end valid_numbers_count_l1070_107087


namespace simplify_expression_l1070_107026

theorem simplify_expression (a b k : ℝ) (h1 : a + b = -k) (h2 : a * b = -3) :
  (a - 3) * (b - 3) = 6 + 3 * k := by
  sorry

end simplify_expression_l1070_107026


namespace expression_evaluation_l1070_107024

theorem expression_evaluation :
  ∀ x : ℕ, 
    x - 3 < 0 →
    x - 1 ≠ 0 →
    x - 2 ≠ 0 →
    (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = 2 :=
by
  sorry

end expression_evaluation_l1070_107024


namespace right_isosceles_not_scalene_l1070_107028

/-- A triangle in Euclidean space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A right isosceles triangle -/
def RightIsosceles (t : Triangle) : Prop :=
  ∃ (a b : ℝ), t.A = (0, 0) ∧ t.B = (a, 0) ∧ t.C = (0, a) ∧ a > 0

/-- A scalene triangle -/
def Scalene (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B ≠ d t.B t.C ∧ d t.B t.C ≠ d t.C t.A ∧ d t.C t.A ≠ d t.A t.B

theorem right_isosceles_not_scalene :
  ∀ t : Triangle, RightIsosceles t → ¬ Scalene t :=
sorry

end right_isosceles_not_scalene_l1070_107028


namespace smith_family_puzzle_l1070_107032

def is_valid_license_plate (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧
  ∃ (a b c : ℕ) (d : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (n.digits 10).count a ≥ 1 ∧
    (n.digits 10).count b ≥ 1 ∧
    (n.digits 10).count c ≥ 1 ∧
    (n.digits 10).count d = 3 ∧
    (n.digits 10).sum = 2 * (n % 100)

theorem smith_family_puzzle :
  ∀ (license_plate : ℕ) (children_ages : List ℕ),
    is_valid_license_plate license_plate →
    children_ages.length = 9 →
    children_ages.maximum = some 10 →
    (∀ age ∈ children_ages, age < 10) →
    (∀ age ∈ children_ages, license_plate % age = 0) →
    4 ∉ children_ages :=
by sorry

end smith_family_puzzle_l1070_107032


namespace valid_sequences_characterization_l1070_107063

/-- Represents the possible weather observations: Plus for no rain, Minus for rain -/
inductive WeatherObservation
| Plus : WeatherObservation
| Minus : WeatherObservation

/-- Represents a sequence of three weather observations -/
structure ObservationSequence :=
  (first : WeatherObservation)
  (second : WeatherObservation)
  (third : WeatherObservation)

/-- Determines if a sequence is valid based on the third student's rule -/
def isValidSequence (seq : ObservationSequence) : Prop :=
  match seq.third with
  | WeatherObservation.Minus => 
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Minus) ∨
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Plus) ∨
      (seq.first = WeatherObservation.Plus ∧ seq.second = WeatherObservation.Minus)
  | WeatherObservation.Plus =>
      (seq.first = WeatherObservation.Plus ∧ seq.second = WeatherObservation.Plus) ∨
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Plus)

/-- The set of all valid observation sequences -/
def validSequences : Set ObservationSequence :=
  { seq | isValidSequence seq }

theorem valid_sequences_characterization :
  validSequences = {
    ⟨WeatherObservation.Plus, WeatherObservation.Plus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Plus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Minus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Minus, WeatherObservation.Minus⟩
  } := by
  sorry

#check valid_sequences_characterization

end valid_sequences_characterization_l1070_107063


namespace remainder_of_binary_division_l1070_107060

-- Define the binary number
def binary_number : ℕ := 101100110011

-- Define the divisor
def divisor : ℕ := 8

-- Theorem statement
theorem remainder_of_binary_division :
  binary_number % divisor = 3 := by
  sorry

end remainder_of_binary_division_l1070_107060


namespace jane_test_probability_l1070_107094

theorem jane_test_probability (pass_prob : ℚ) (h : pass_prob = 4/7) :
  1 - pass_prob = 3/7 := by
  sorry

end jane_test_probability_l1070_107094


namespace M_divisible_by_41_l1070_107005

def M : ℕ := sorry

theorem M_divisible_by_41 : 41 ∣ M := by sorry

end M_divisible_by_41_l1070_107005


namespace fish_estimation_result_l1070_107089

/-- Represents the catch-release-recatch method for estimating fish population -/
structure FishEstimation where
  initial_catch : ℕ
  initial_marked : ℕ
  second_catch : ℕ
  second_marked : ℕ

/-- Calculates the estimated number of fish in the pond -/
def estimate_fish_population (fe : FishEstimation) : ℕ :=
  (fe.initial_marked * fe.second_catch) / fe.second_marked

/-- Theorem stating that the estimated number of fish in the pond is 2500 -/
theorem fish_estimation_result :
  let fe : FishEstimation := {
    initial_catch := 100,
    initial_marked := 100,
    second_catch := 200,
    second_marked := 8
  }
  estimate_fish_population fe = 2500 := by
  sorry


end fish_estimation_result_l1070_107089


namespace min_value_of_f_l1070_107098

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem min_value_of_f :
  ∃ (x_min : ℝ), (∀ x, f x ≥ f x_min) ∧ (x_min = -1) ∧ (f x_min = 3) :=
sorry

end min_value_of_f_l1070_107098


namespace gcd_factorial_8_and_cube_factorial_6_l1070_107056

theorem gcd_factorial_8_and_cube_factorial_6 :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 6 ^ 3) = 11520 := by
  sorry

end gcd_factorial_8_and_cube_factorial_6_l1070_107056


namespace shooting_competition_probabilities_l1070_107042

/-- Probability of A hitting the target in a single shot -/
def prob_A_hit : ℚ := 2/3

/-- Probability of B hitting the target in a single shot -/
def prob_B_hit : ℚ := 3/4

/-- Number of consecutive shots -/
def num_shots : ℕ := 3

theorem shooting_competition_probabilities :
  let prob_A_miss_at_least_once := 1 - prob_A_hit ^ num_shots
  let prob_A_hit_twice := (num_shots.choose 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)
  let prob_B_hit_once := (num_shots.choose 1 : ℚ) * prob_B_hit * (1 - prob_B_hit)^2
  prob_A_miss_at_least_once = 19/27 ∧
  prob_A_hit_twice * prob_B_hit_once = 1/16 := by
  sorry


end shooting_competition_probabilities_l1070_107042


namespace victor_trays_capacity_l1070_107002

/-- The number of trays Victor picked up from the first table -/
def trays_table1 : ℕ := 23

/-- The number of trays Victor picked up from the second table -/
def trays_table2 : ℕ := 5

/-- The total number of trips Victor made -/
def total_trips : ℕ := 4

/-- The number of trays Victor could carry at a time -/
def trays_per_trip : ℕ := (trays_table1 + trays_table2) / total_trips

theorem victor_trays_capacity : trays_per_trip = 7 := by
  sorry

end victor_trays_capacity_l1070_107002


namespace four_digit_divisible_by_9_l1070_107055

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem four_digit_divisible_by_9 (B : ℕ) :
  B < 10 →
  is_divisible_by_9 (4000 + 100 * B + 10 * B + 2) →
  B = 6 := by
  sorry

#check four_digit_divisible_by_9

end four_digit_divisible_by_9_l1070_107055


namespace complex_magnitude_l1070_107019

theorem complex_magnitude (w z : ℂ) 
  (h1 : w * z + 2 * w - 3 * z = 10 - 6 * Complex.I)
  (h2 : Complex.abs w = 2)
  (h3 : Complex.abs (w + 2) = 3) :
  Complex.abs z = (2 * Real.sqrt 34 - 4) / 5 := by
sorry

end complex_magnitude_l1070_107019


namespace point_coordinates_wrt_origin_l1070_107001

/-- The coordinates of a point P with respect to the origin are the same as its given coordinates in a Cartesian coordinate system. -/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  P = (x, y) := by sorry

end point_coordinates_wrt_origin_l1070_107001


namespace union_of_sets_l1070_107035

def set_A : Set ℝ := {x | x^2 - x = 0}
def set_B : Set ℝ := {x | x^2 + x = 0}

theorem union_of_sets : set_A ∪ set_B = {-1, 0, 1} := by sorry

end union_of_sets_l1070_107035


namespace prob_specific_draw_l1070_107027

def standard_deck : ℕ := 52

def prob_first_five (deck : ℕ) : ℚ := 4 / deck

def prob_second_diamond (deck : ℕ) : ℚ := 13 / (deck - 1)

def prob_third_three (deck : ℕ) : ℚ := 4 / (deck - 2)

theorem prob_specific_draw (deck : ℕ) (h : deck = standard_deck) :
  prob_first_five deck * prob_second_diamond deck * prob_third_three deck = 17 / 11050 := by
  sorry

end prob_specific_draw_l1070_107027


namespace solve_for_A_l1070_107031

theorem solve_for_A : ∃ A : ℝ, 4 * A + 5 = 33 ∧ A = 7 := by
  sorry

end solve_for_A_l1070_107031


namespace cheryl_material_usage_l1070_107053

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 5 / 11)
  (h2 : material2 = 2 / 3)
  (h3 : leftover = 25 / 55) :
  material1 + material2 - leftover = 22 / 33 := by
  sorry

end cheryl_material_usage_l1070_107053


namespace sequence_zero_l1070_107091

/-- A sequence of real numbers indexed by positive integers. -/
def RealSequence := ℕ+ → ℝ

/-- The property that b_n ≤ c_n for all n. -/
def LessEqProperty (b c : RealSequence) : Prop :=
  ∀ n : ℕ+, b n ≤ c n

/-- The property that b_{n+1} and c_{n+1} are roots of x^2 + b_n*x + c_n = 0. -/
def RootProperty (b c : RealSequence) : Prop :=
  ∀ n : ℕ+, (b (n + 1))^2 + (b n) * (b (n + 1)) + (c n) = 0 ∧
            (c (n + 1))^2 + (b n) * (c (n + 1)) + (c n) = 0

theorem sequence_zero (b c : RealSequence) 
  (h1 : LessEqProperty b c) (h2 : RootProperty b c) :
  (∀ n : ℕ+, b n = 0 ∧ c n = 0) :=
sorry

end sequence_zero_l1070_107091


namespace gcd_of_35_91_840_l1070_107057

theorem gcd_of_35_91_840 : Nat.gcd 35 (Nat.gcd 91 840) = 7 := by
  sorry

end gcd_of_35_91_840_l1070_107057


namespace distance_range_l1070_107025

/-- Hyperbola C with equation x^2 - y^2/3 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Focal length of the hyperbola -/
def focal_length : ℝ := 4

/-- Right triangle ABD formed by intersection with perpendicular line through right focus -/
def right_triangle_ABD : Prop := sorry

/-- Slopes of lines AM and AN -/
def slope_product (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -2

/-- Distance from A to line MN -/
def distance_A_to_MN (d : ℝ) : Prop := sorry

/-- Theorem stating the range of distance d -/
theorem distance_range :
  ∀ (d : ℝ), hyperbola_C 1 0 →
  focal_length = 4 →
  right_triangle_ABD →
  (∃ k₁ k₂, slope_product k₁ k₂) →
  distance_A_to_MN d →
  3 * Real.sqrt 3 < d ∧ d ≤ 6 := by sorry

end distance_range_l1070_107025


namespace percentage_problem_l1070_107000

theorem percentage_problem (n : ℝ) (p : ℝ) : 
  n = 50 → 
  p / 100 * n = 30 / 100 * 10 + 27 → 
  p = 60 := by
sorry

end percentage_problem_l1070_107000


namespace rhombus_diagonal_length_l1070_107082

/-- Proves that in a rhombus with an area of 127.5 cm² and one diagonal of 15 cm, 
    the length of the other diagonal is 17 cm. -/
theorem rhombus_diagonal_length (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 127.5 → d1 = 15 → area = (d1 * d2) / 2 → d2 = 17 := by
  sorry

end rhombus_diagonal_length_l1070_107082


namespace sin_double_alpha_l1070_107046

theorem sin_double_alpha (α : Real) 
  (h : Real.cos (α - Real.pi/4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = -3/4 := by
sorry

end sin_double_alpha_l1070_107046


namespace red_card_events_l1070_107061

-- Define the set of colors
inductive Color : Type
| Red | Black | White | Blue

-- Define the set of individuals
inductive Person : Type
| A | B | C | D

-- Define a distribution as a function from Person to Color
def Distribution := Person → Color

-- Define the event "A receives the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Color.Red

-- Define the event "B receives the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Color.Red

-- Theorem: A_gets_red and B_gets_red are mutually exclusive but not complementary
theorem red_card_events (d : Distribution) :
  (¬ (A_gets_red d ∧ B_gets_red d)) ∧
  (∃ (d : Distribution), ¬ A_gets_red d ∧ ¬ B_gets_red d) :=
by sorry

end red_card_events_l1070_107061


namespace lincoln_county_houses_l1070_107018

/-- The original number of houses in Lincoln County -/
def original_houses : ℕ := 20817

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 97741

/-- The current number of houses in Lincoln County -/
def current_houses : ℕ := 118558

/-- Theorem stating that the original number of houses plus the houses built
    during the boom equals the current number of houses -/
theorem lincoln_county_houses :
  original_houses + houses_built = current_houses := by
  sorry

end lincoln_county_houses_l1070_107018


namespace second_train_length_l1070_107099

/-- Calculates the length of the second train given the parameters of two trains approaching each other. -/
theorem second_train_length 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (clear_time : ℝ) 
  (h1 : length_train1 = 120)
  (h2 : speed_train1 = 42)
  (h3 : speed_train2 = 30)
  (h4 : clear_time = 20.99832013438925) :
  ∃ (length_train2 : ℝ), 
    abs (length_train2 - 299.97) < 0.01 ∧ 
    length_train1 + length_train2 = (speed_train1 + speed_train2) * (1000 / 3600) * clear_time := by
  sorry

end second_train_length_l1070_107099


namespace value_of_y_l1070_107034

theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end value_of_y_l1070_107034


namespace arithmetic_progression_special_case_l1070_107007

/-- 
Given an arithmetic progression (a_n) where a_k = l and a_l = k (k ≠ l),
prove that the general term a_n is equal to k + l - n.
-/
theorem arithmetic_progression_special_case 
  (a : ℕ → ℤ) (k l : ℕ) (h_neq : k ≠ l) 
  (h_arith : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_k : a k = l) (h_l : a l = k) :
  ∀ n : ℕ, a n = k + l - n :=
sorry

end arithmetic_progression_special_case_l1070_107007


namespace initial_figures_correct_figure_50_l1070_107065

/-- The number of unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The first four figures match the given pattern -/
theorem initial_figures_correct :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 := by sorry

/-- The 50th figure contains 7651 unit squares -/
theorem figure_50 : f 50 = 7651 := by sorry

end initial_figures_correct_figure_50_l1070_107065


namespace parallel_vectors_x_value_l1070_107052

def vector_a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![3, x + 1]

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ vector_a x = k • vector_b x) → x = 2 := by
sorry

end parallel_vectors_x_value_l1070_107052


namespace mod_thirteen_equiv_l1070_107062

theorem mod_thirteen_equiv (n : ℤ) : 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2345 [ZMOD 13] → n = 8 := by
  sorry

end mod_thirteen_equiv_l1070_107062


namespace parallelogram_area_l1070_107040

theorem parallelogram_area (a b : ℝ) (θ : ℝ) 
  (ha : a = 20) (hb : b = 10) (hθ : θ = 150 * π / 180) : 
  a * b * Real.sin θ = 100 := by
  sorry

end parallelogram_area_l1070_107040


namespace expression_evaluation_l1070_107029

theorem expression_evaluation : 23 - 17 - (-7) + (-16) = -3 := by
  sorry

end expression_evaluation_l1070_107029


namespace complement_of_union_sets_l1070_107015

open Set

theorem complement_of_union_sets (A B : Set ℝ) :
  A = {x : ℝ | x < 1} →
  B = {x : ℝ | x > 3} →
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_of_union_sets_l1070_107015


namespace no_unique_solution_l1070_107033

theorem no_unique_solution (x y z : ℕ+) : 
  ¬∃ (f : ℕ+ → ℕ+ → ℕ+ → Prop), 
    (∀ (a b c : ℕ+), f a b c ↔ Real.sqrt (a^2 + Real.sqrt ((b:ℝ)/(c:ℝ))) = (b:ℝ)^2 * Real.sqrt ((a:ℝ)/(c:ℝ))) ∧
    (∃! (g : ℕ+ → ℕ+ → ℕ+ → Prop), ∀ (a b c : ℕ+), g a b c ↔ f a b c) :=
sorry

end no_unique_solution_l1070_107033


namespace initial_to_doubled_ratio_l1070_107030

theorem initial_to_doubled_ratio (x : ℝ) (h : 3 * (2 * x + 13) = 93) : 
  x / (2 * x) = 1 / 2 := by
sorry

end initial_to_doubled_ratio_l1070_107030


namespace converse_opposites_sum_zero_l1070_107009

theorem converse_opposites_sum_zero :
  ∀ x y : ℝ, (x = -y) → (x + y = 0) := by
  sorry

end converse_opposites_sum_zero_l1070_107009


namespace sin_240_degrees_l1070_107081

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l1070_107081


namespace sum_of_five_consecutive_squares_not_perfect_square_l1070_107045

theorem sum_of_five_consecutive_squares_not_perfect_square (x : ℤ) : 
  ¬∃ (k : ℤ), 5 * (x^2 + 2) = k^2 := by
sorry

end sum_of_five_consecutive_squares_not_perfect_square_l1070_107045


namespace expression_simplification_l1070_107080

/-- Proves that the simplification of 7y + 8 - 3y + 15 + 2x is equivalent to 4y + 2x + 23 -/
theorem expression_simplification (x y : ℝ) :
  7 * y + 8 - 3 * y + 15 + 2 * x = 4 * y + 2 * x + 23 := by
  sorry

end expression_simplification_l1070_107080


namespace student_multiplication_error_l1070_107017

theorem student_multiplication_error (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  (78 : ℚ) * ((1 + (100 * a + 10 * b + c : ℚ) / 999) - (1 + (a / 10 + b / 100 + c / 1000))) = (3 / 5) →
  100 * a + 10 * b + c = 765 := by
  sorry

end student_multiplication_error_l1070_107017


namespace part1_part2_part3_l1070_107043

-- Define the functions f and g
def f (x : ℝ) := x - 2
def g (m : ℝ) (x : ℝ) := x^2 - 2*m*x + 4

-- Part 1
theorem part1 (m : ℝ) :
  (∀ x, g m x > f x) ↔ m ∈ Set.Ioo (-Real.sqrt 6 - 1/2) (Real.sqrt 6 - 1/2) :=
sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 4 5, g m x₁ = f x₂) ↔ 
  m ∈ Set.Icc (5/4) (Real.sqrt 2) :=
sorry

-- Part 3
theorem part3 :
  (∀ n : ℝ, ∃ x₀ ∈ Set.Icc (-2) 2, |g (-1) x₀ - x₀^2 + n| ≥ k) ↔
  k ∈ Set.Iic 4 :=
sorry

end part1_part2_part3_l1070_107043


namespace f_monotone_decreasing_min_a_value_l1070_107058

noncomputable section

def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

def g (x : ℝ) := x * Real.exp (-x)

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def monotone_decreasing_intervals (f : ℝ → ℝ) : Set (Set ℝ) :=
  {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) ∧
    is_monotone_decreasing f (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)}

theorem f_monotone_decreasing : 
  monotone_decreasing_intervals f = {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)} :=
sorry

theorem min_a_value :
  (∃ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 0 (Real.pi / 2) → 
    g x₁ + a + 3 > f x₂) ∧
  (∀ a' : ℝ, a' < -3 / Real.exp 3 → 
    ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
      g x₁ + a' + 3 ≤ f x₂) :=
sorry

end f_monotone_decreasing_min_a_value_l1070_107058


namespace louise_pencils_l1070_107023

theorem louise_pencils (box_capacity : ℕ) (red_pencils : ℕ) (yellow_pencils : ℕ) (total_boxes : ℕ)
  (h1 : box_capacity = 20)
  (h2 : red_pencils = 20)
  (h3 : yellow_pencils = 40)
  (h4 : total_boxes = 8) :
  let blue_pencils := 2 * red_pencils
  let total_capacity := box_capacity * total_boxes
  let other_pencils := red_pencils + blue_pencils + yellow_pencils
  let green_pencils := total_capacity - other_pencils
  green_pencils = 60 ∧ green_pencils = red_pencils + blue_pencils :=
by sorry


end louise_pencils_l1070_107023


namespace jessica_cut_four_orchids_l1070_107084

/-- The number of orchids Jessica cut from her garden -/
def orchids_cut (initial_orchids final_orchids : ℕ) : ℕ :=
  final_orchids - initial_orchids

/-- Theorem stating that Jessica cut 4 orchids -/
theorem jessica_cut_four_orchids :
  orchids_cut 3 7 = 4 := by
  sorry

end jessica_cut_four_orchids_l1070_107084


namespace inhabitable_earth_surface_l1070_107016

theorem inhabitable_earth_surface (total_surface area_land area_inhabitable : ℝ) :
  area_land = (1 / 3 : ℝ) * total_surface →
  area_inhabitable = (3 / 4 : ℝ) * area_land →
  area_inhabitable = (1 / 4 : ℝ) * total_surface :=
by
  sorry

end inhabitable_earth_surface_l1070_107016


namespace tenth_term_is_one_over_120_l1070_107014

def a (n : ℕ) : ℚ := 1 / (n * (n + 2))

theorem tenth_term_is_one_over_120 : a 10 = 1 / 120 := by
  sorry

end tenth_term_is_one_over_120_l1070_107014


namespace smallest_n_for_integer_sum_eighteen_makes_sum_integer_smallest_n_is_eighteen_l1070_107079

theorem smallest_n_for_integer_sum : 
  ∀ n : ℕ+, (1/3 + 1/4 + 1/9 + 1/n : ℚ).isInt → n ≥ 18 := by
  sorry

theorem eighteen_makes_sum_integer : 
  (1/3 + 1/4 + 1/9 + 1/18 : ℚ).isInt := by
  sorry

theorem smallest_n_is_eighteen : 
  ∃! n : ℕ+, (1/3 + 1/4 + 1/9 + 1/n : ℚ).isInt ∧ ∀ m : ℕ+, (1/3 + 1/4 + 1/9 + 1/m : ℚ).isInt → n ≤ m := by
  sorry

end smallest_n_for_integer_sum_eighteen_makes_sum_integer_smallest_n_is_eighteen_l1070_107079


namespace area_triangle_ABC_in_special_cyclic_quadrilateral_l1070_107085

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Checks if a quadrilateral is cyclic (inscribed in a circle) -/
def isCyclic (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point := sorry

/-- Theorem: Area of triangle ABC in a special cyclic quadrilateral -/
theorem area_triangle_ABC_in_special_cyclic_quadrilateral 
  (A B C D E : Point) (c : Circle) :
  isCyclic ⟨A, B, C, D⟩ c →
  E = intersectionPoint A C B D →
  A.x = D.x ∧ A.y = D.y →
  (C.x - E.x) / (E.x - D.x) = 3 / 2 ∧ (C.y - E.y) / (E.y - D.y) = 3 / 2 →
  triangleArea A B E = 8 →
  triangleArea A B C = 18 := by sorry

end area_triangle_ABC_in_special_cyclic_quadrilateral_l1070_107085


namespace base_three_20121_equals_178_l1070_107039

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [2, 0, 1, 2, 1] = 178 := by
  sorry

end base_three_20121_equals_178_l1070_107039


namespace inequality_problem_l1070_107008

theorem inequality_problem (m n : ℝ) (h : ∀ x : ℝ, m * x^2 + n * x - 1/m < 0 ↔ x < -1/2 ∨ x > 2) :
  (m = -1 ∧ n = 3/2) ∧
  (∀ a : ℝ, 
    (a < 1 → ∀ x : ℝ, (2*a-1-x)*(x+m) > 0 ↔ 2*a-1 < x ∧ x < 1) ∧
    (a = 1 → ∀ x : ℝ, ¬((2*a-1-x)*(x+m) > 0)) ∧
    (a > 1 → ∀ x : ℝ, (2*a-1-x)*(x+m) > 0 ↔ 1 < x ∧ x < 2*a-1)) :=
by sorry

end inequality_problem_l1070_107008


namespace roulette_wheel_probability_l1070_107090

/-- The probability of a roulette wheel landing on section F -/
def prob_F (prob_D prob_E prob_G : ℚ) : ℚ :=
  1 - (prob_D + prob_E + prob_G)

/-- Theorem: The probability of landing on section F is 1/4 -/
theorem roulette_wheel_probability :
  let prob_D : ℚ := 3/8
  let prob_E : ℚ := 1/4
  let prob_G : ℚ := 1/8
  prob_F prob_D prob_E prob_G = 1/4 := by
  sorry

end roulette_wheel_probability_l1070_107090


namespace unique_solution_l1070_107021

/-- Define a sequence of 100 real numbers satisfying given conditions -/
def SequenceOfHundred (a : Fin 100 → ℝ) : Prop :=
  (∀ i : Fin 99, a i - 4 * a (i + 1) + 3 * a (i + 2) ≥ 0) ∧
  (a 99 - 4 * a 0 + 3 * a 1 ≥ 0) ∧
  (a 0 = 1)

/-- Theorem stating that the sequence of all 1's is the unique solution -/
theorem unique_solution (a : Fin 100 → ℝ) (h : SequenceOfHundred a) :
  ∀ i : Fin 100, a i = 1 := by
  sorry

end unique_solution_l1070_107021


namespace complex_power_magnitude_l1070_107074

theorem complex_power_magnitude : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end complex_power_magnitude_l1070_107074


namespace range_of_m_l1070_107011

/-- α is the condition that x ≤ -5 or x ≥ 1 -/
def α (x : ℝ) : Prop := x ≤ -5 ∨ x ≥ 1

/-- β is the condition that 2m-3 ≤ x ≤ 2m+1 -/
def β (m x : ℝ) : Prop := 2*m - 3 ≤ x ∧ x ≤ 2*m + 1

/-- α is a necessary condition for β -/
def α_necessary_for_β (m : ℝ) : Prop := ∀ x, β m x → α x

theorem range_of_m (m : ℝ) : α_necessary_for_β m → m ≥ 2 ∨ m ≤ -3 := by
  sorry

end range_of_m_l1070_107011


namespace same_solution_implies_c_value_l1070_107068

theorem same_solution_implies_c_value (c : ℝ) :
  (∃ x : ℝ, 3 * x + 8 = 5 ∧ c * x + 15 = 3) → c = 12 := by
  sorry

end same_solution_implies_c_value_l1070_107068


namespace perfect_negative_correlation_l1070_107006

/-- Represents a pair of data points (x, y) -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Calculates the sample correlation coefficient for a list of data points -/
def sampleCorrelationCoefficient (data : List DataPoint) : ℝ :=
  sorry

/-- Theorem: For any set of paired sample data that fall on a straight line with negative slope,
    the sample correlation coefficient is -1 -/
theorem perfect_negative_correlation 
  (data : List DataPoint) 
  (h_line : ∃ (m : ℝ) (b : ℝ), m < 0 ∧ ∀ (point : DataPoint), point ∈ data → point.y = m * point.x + b) :
  sampleCorrelationCoefficient data = -1 :=
sorry

end perfect_negative_correlation_l1070_107006


namespace darla_electricity_bill_l1070_107083

-- Define the cost per watt in cents
def cost_per_watt : ℕ := 400

-- Define the amount of electricity used in watts
def electricity_used : ℕ := 300

-- Define the late fee in cents
def late_fee : ℕ := 15000

-- Define the total cost in cents
def total_cost : ℕ := cost_per_watt * electricity_used + late_fee

-- Theorem statement
theorem darla_electricity_bill : total_cost = 135000 := by
  sorry

end darla_electricity_bill_l1070_107083


namespace book_selection_theorem_l1070_107047

theorem book_selection_theorem (chinese_books math_books sports_books : ℕ) 
  (h1 : chinese_books = 4) 
  (h2 : math_books = 5) 
  (h3 : sports_books = 6) : 
  (chinese_books + math_books + sports_books = 15) ∧ 
  (chinese_books * math_books * sports_books = 120) := by
  sorry

end book_selection_theorem_l1070_107047


namespace power_fraction_equality_l1070_107077

theorem power_fraction_equality : (88888 ^ 5 : ℚ) / (22222 ^ 5) = 1024 := by
  sorry

end power_fraction_equality_l1070_107077


namespace island_marriage_proportion_l1070_107097

theorem island_marriage_proportion (men women : ℕ) (h1 : 2 * men = 3 * women) :
  (2 * men + 2 * women : ℚ) / (3 * men + 5 * women : ℚ) = 12 / 19 := by
  sorry

end island_marriage_proportion_l1070_107097


namespace more_heads_than_tails_probability_l1070_107076

/-- The probability of getting more heads than tails when tossing a fair coin 4 times -/
def probability_more_heads_than_tails : ℚ := 5/16

/-- A fair coin is tossed 4 times -/
def num_tosses : ℕ := 4

/-- The probability of getting heads on a single toss of a fair coin -/
def probability_heads : ℚ := 1/2

/-- The probability of getting tails on a single toss of a fair coin -/
def probability_tails : ℚ := 1/2

theorem more_heads_than_tails_probability :
  probability_more_heads_than_tails = 
    (Nat.choose num_tosses 3 : ℚ) * probability_heads^3 * probability_tails +
    (Nat.choose num_tosses 4 : ℚ) * probability_heads^4 :=
by sorry

end more_heads_than_tails_probability_l1070_107076


namespace rectangle_area_in_isosceles_triangle_l1070_107036

/-- The area of a rectangle inscribed in an isosceles triangle -/
theorem rectangle_area_in_isosceles_triangle 
  (b h x : ℝ) 
  (hb : b > 0) 
  (hh : h > 0) 
  (hx : x > 0) 
  (hx_bound : x < h/2) : 
  let rectangle_area := x * (b/2 - b*x/h)
  ∃ (rectangle_base : ℝ), 
    rectangle_base > 0 ∧ 
    rectangle_base = b * (h/2 - x) / h ∧
    rectangle_area = x * rectangle_base :=
by sorry

end rectangle_area_in_isosceles_triangle_l1070_107036


namespace intersection_is_empty_l1070_107004

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l1070_107004


namespace function_inequality_implies_a_bound_l1070_107049

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 → 
    x₁^3 - 3*x₁ ≤ Real.exp x₂ - 2*a*x₂ + 2) → 
  a ≤ Real.exp 1 / 2 := by
  sorry

end function_inequality_implies_a_bound_l1070_107049


namespace quadratic_zero_discriminant_geometric_progression_l1070_107038

/-- 
Given a quadratic equation ax^2 + 3bx + c = 0 with zero discriminant,
prove that the coefficients a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) (h_nonzero : a ≠ 0) 
  (h_discriminant : 9 * b^2 - 4 * a * c = 0) :
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r :=
sorry

end quadratic_zero_discriminant_geometric_progression_l1070_107038


namespace initial_kola_solution_volume_l1070_107066

/-- Represents the composition and volume of a kola solution -/
structure KolaSolution where
  initialVolume : ℝ
  waterPercentage : ℝ
  concentratedKolaPercentage : ℝ
  sugarPercentage : ℝ

/-- Theorem stating the initial volume of the kola solution -/
theorem initial_kola_solution_volume
  (solution : KolaSolution)
  (h1 : solution.waterPercentage = 0.88)
  (h2 : solution.concentratedKolaPercentage = 0.05)
  (h3 : solution.sugarPercentage = 1 - solution.waterPercentage - solution.concentratedKolaPercentage)
  (h4 : let newVolume := solution.initialVolume + 3.2 + 10 + 6.8
        (solution.sugarPercentage * solution.initialVolume + 3.2) / newVolume = 0.075) :
  solution.initialVolume = 340 := by
  sorry

end initial_kola_solution_volume_l1070_107066


namespace optimal_speed_yihuang_expressway_l1070_107003

/-- The optimal speed problem for the Yihuang Expressway -/
theorem optimal_speed_yihuang_expressway 
  (total_length : ℝ) 
  (min_speed max_speed : ℝ) 
  (fixed_cost : ℝ) 
  (k : ℝ) 
  (max_total_cost : ℝ) :
  total_length = 350 →
  min_speed = 60 →
  max_speed = 120 →
  fixed_cost = 200 →
  k * max_speed^2 + fixed_cost = max_total_cost →
  max_total_cost = 488 →
  ∃ (optimal_speed : ℝ), 
    optimal_speed = 100 ∧
    ∀ (v : ℝ), min_speed ≤ v ∧ v ≤ max_speed →
      total_length * (fixed_cost / v + k * v) ≥ 
      total_length * (fixed_cost / optimal_speed + k * optimal_speed) :=
by sorry

end optimal_speed_yihuang_expressway_l1070_107003


namespace piecewise_continuity_l1070_107037

/-- A piecewise function f defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x^2 + x + 2 else 2*x + a

/-- Theorem stating that the piecewise function f is continuous at x = 3 if and only if a = 8 -/
theorem piecewise_continuity (a : ℝ) :
  ContinuousAt (f a) 3 ↔ a = 8 := by sorry

end piecewise_continuity_l1070_107037


namespace cube_root_problem_l1070_107048

theorem cube_root_problem (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := by
  sorry

end cube_root_problem_l1070_107048


namespace complement_A_intersect_B_l1070_107012

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- State the theorem
theorem complement_A_intersect_B : (Set.compl A) ∩ B = Set.Ioi 1 := by
  sorry

end complement_A_intersect_B_l1070_107012


namespace rectangular_prism_diagonal_l1070_107010

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end rectangular_prism_diagonal_l1070_107010


namespace a_range_l1070_107070

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else x^2 + 4/x + a * Real.log x

/-- The theorem statement -/
theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  2 ≤ a ∧ a ≤ 5 :=
sorry

end a_range_l1070_107070


namespace solve_exponential_equation_l1070_107013

theorem solve_exponential_equation :
  ∃ y : ℝ, (1000 : ℝ)^4 = 10^y ↔ y = 12 := by
  sorry

end solve_exponential_equation_l1070_107013


namespace jellybean_count_l1070_107095

/-- The number of jellybeans in a bag with specific color distributions -/
def total_jellybeans (black green orange red yellow : ℕ) : ℕ :=
  black + green + orange + red + yellow

/-- Theorem stating the total number of jellybeans in the bag -/
theorem jellybean_count : ∃ (black green orange red yellow : ℕ),
  black = 8 ∧
  green = black + 4 ∧
  orange = green - 5 ∧
  red = orange + 3 ∧
  yellow = black - 2 ∧
  total_jellybeans black green orange red yellow = 43 := by
  sorry


end jellybean_count_l1070_107095


namespace inequality_solution_l1070_107071

theorem inequality_solution (a : ℝ) (h : |a + 1| < 3) :
  (∀ x, x - (a + 1) * (x + 1) > 0 ↔ 
    ((-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨
     (a = -2 ∧ x ≠ -1) ∨
     (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)))) :=
by sorry

end inequality_solution_l1070_107071


namespace age_difference_l1070_107064

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : a = c + 10 := by
  sorry

end age_difference_l1070_107064


namespace cory_fruit_eating_orders_l1070_107041

/-- Represents the number of fruits Cory has of each type -/
structure FruitInventory where
  apples : Nat
  bananas : Nat
  mangoes : Nat

/-- Represents the constraints of Cory's fruit-eating schedule -/
structure EatingSchedule where
  days : Nat
  startsWithApple : Bool
  endsWithApple : Bool

/-- Calculates the number of ways Cory can eat his fruits given his inventory and schedule constraints -/
def countEatingOrders (inventory : FruitInventory) (schedule : EatingSchedule) : Nat :=
  sorry

/-- Theorem stating that given Cory's specific fruit inventory and eating schedule, 
    there are exactly 80 different orders in which he can eat his fruits -/
theorem cory_fruit_eating_orders :
  let inventory : FruitInventory := ⟨3, 3, 1⟩
  let schedule : EatingSchedule := ⟨7, true, true⟩
  countEatingOrders inventory schedule = 80 :=
by sorry

end cory_fruit_eating_orders_l1070_107041


namespace square_sum_value_l1070_107054

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := by
  sorry

end square_sum_value_l1070_107054


namespace stick_difference_l1070_107050

/-- 
Given:
- Dave picked up 14 sticks
- Amy picked up 9 sticks
- There were initially 50 sticks in the yard

Prove that the difference between the number of sticks picked up by Dave and Amy
and the number of sticks left in the yard is 4.
-/
theorem stick_difference (dave_sticks amy_sticks initial_sticks : ℕ) 
  (h1 : dave_sticks = 14)
  (h2 : amy_sticks = 9)
  (h3 : initial_sticks = 50) :
  let picked_up := dave_sticks + amy_sticks
  let left_in_yard := initial_sticks - picked_up
  picked_up - left_in_yard = 4 := by
  sorry

end stick_difference_l1070_107050


namespace square_plus_abs_zero_implies_both_zero_l1070_107044

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) :
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
sorry

end square_plus_abs_zero_implies_both_zero_l1070_107044


namespace machine_A_rate_l1070_107069

/-- Production rates of machines A, P, and Q -/
structure MachineRates where
  rateA : ℝ
  rateP : ℝ
  rateQ : ℝ

/-- Time taken by machines P and Q to produce 220 sprockets -/
structure MachineTimes where
  timeP : ℝ
  timeQ : ℝ

/-- Conditions of the sprocket manufacturing problem -/
def sprocketProblem (r : MachineRates) (t : MachineTimes) : Prop :=
  220 / t.timeP = r.rateP
  ∧ 220 / t.timeQ = r.rateQ
  ∧ t.timeP = t.timeQ + 10
  ∧ r.rateQ = 1.1 * r.rateA
  ∧ r.rateA > 0
  ∧ r.rateP > 0
  ∧ r.rateQ > 0
  ∧ t.timeP > 0
  ∧ t.timeQ > 0

/-- Theorem stating that machine A's production rate is 20/9 sprockets per hour -/
theorem machine_A_rate (r : MachineRates) (t : MachineTimes) 
  (h : sprocketProblem r t) : r.rateA = 20/9 := by
  sorry

end machine_A_rate_l1070_107069


namespace circle_center_l1070_107059

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenCircleEquation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y = 16

theorem circle_center :
  ∃ r, ∀ x y, GivenCircleEquation x y ↔ CircleEquation 4 2 r x y :=
sorry

end circle_center_l1070_107059
