import Mathlib

namespace bike_price_proof_l37_3767

theorem bike_price_proof (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percentage = 0.20 →
  upfront_payment = 240 →
  upfront_percentage * total_price = upfront_payment →
  total_price = 1200 := by
  sorry

end bike_price_proof_l37_3767


namespace acme_cheaper_at_min_shirts_l37_3763

/-- Acme T-Shirt Company's setup fee -/
def acme_setup : ℕ := 60

/-- Acme T-Shirt Company's per-shirt cost -/
def acme_per_shirt : ℕ := 11

/-- Gamma T-Shirt Company's setup fee -/
def gamma_setup : ℕ := 10

/-- Gamma T-Shirt Company's per-shirt cost -/
def gamma_per_shirt : ℕ := 16

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts_acme_cheaper <
  gamma_setup + gamma_per_shirt * min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper →
    acme_setup + acme_per_shirt * n ≥ gamma_setup + gamma_per_shirt * n :=
by sorry

end acme_cheaper_at_min_shirts_l37_3763


namespace compute_expression_l37_3721

theorem compute_expression : 12 + 10 * (4 - 9)^2 = 262 := by
  sorry

end compute_expression_l37_3721


namespace scale_length_theorem_l37_3723

/-- A scale divided into equal parts -/
structure Scale where
  num_parts : ℕ
  part_length : ℝ

/-- The total length of a scale -/
def total_length (s : Scale) : ℝ := s.num_parts * s.part_length

/-- Theorem stating that a scale with 2 parts of 40 inches each has a total length of 80 inches -/
theorem scale_length_theorem :
  ∀ (s : Scale), s.num_parts = 2 ∧ s.part_length = 40 → total_length s = 80 :=
by
  sorry

end scale_length_theorem_l37_3723


namespace first_hundred_complete_l37_3762

/-- Represents a color of a number in the sequence -/
inductive Color
| Blue
| Red

/-- Represents the properties of the sequence of 200 numbers -/
structure NumberSequence :=
  (numbers : Fin 200 → ℕ)
  (colors : Fin 200 → Color)
  (blue_ascending : ∀ i j, i < j → colors i = Color.Blue → colors j = Color.Blue → numbers i < numbers j)
  (red_descending : ∀ i j, i < j → colors i = Color.Red → colors j = Color.Red → numbers i > numbers j)
  (blue_range : ∀ n, n ∈ Finset.range 100 → ∃ i, colors i = Color.Blue ∧ numbers i = n + 1)
  (red_range : ∀ n, n ∈ Finset.range 100 → ∃ i, colors i = Color.Red ∧ numbers i = 100 - n)

/-- The main theorem stating that the first 100 numbers contain all natural numbers from 1 to 100 -/
theorem first_hundred_complete (seq : NumberSequence) :
  ∀ n, n ∈ Finset.range 100 → ∃ i, i < 100 ∧ seq.numbers i = n + 1 :=
sorry

end first_hundred_complete_l37_3762


namespace baseball_team_wins_l37_3709

theorem baseball_team_wins (total_games : ℕ) (wins losses : ℕ) : 
  total_games = 130 →
  wins + losses = total_games →
  wins = 3 * losses + 14 →
  wins = 101 := by
sorry

end baseball_team_wins_l37_3709


namespace range_of_a_l37_3749

-- Define the sets S and T
def S : Set ℝ := {x | x < -1 ∨ x > 5}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (S ∪ T a = Set.univ) → (-3 < a ∧ a < -1) :=
by sorry

end range_of_a_l37_3749


namespace sqrt_product_equality_l37_3769

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l37_3769


namespace min_value_of_expression_l37_3726

theorem min_value_of_expression (x y : ℝ) : 
  (x*y - 2)^2 + (x - 1 + y)^2 ≥ 2 ∧ 
  ∃ (a b : ℝ), (a*b - 2)^2 + (a - 1 + b)^2 = 2 :=
by sorry


end min_value_of_expression_l37_3726


namespace a_is_irrational_l37_3720

/-- The n-th digit after the decimal point of a real number -/
noncomputable def nthDigitAfterDecimal (a : ℝ) (n : ℕ) : ℕ := sorry

/-- The digit to the left of the decimal point of a real number -/
noncomputable def digitLeftOfDecimal (x : ℝ) : ℕ := sorry

/-- A real number a satisfying the given condition -/
noncomputable def a : ℝ := sorry

/-- The condition that relates a to √2 -/
axiom a_condition : ∀ n : ℕ, nthDigitAfterDecimal a n = digitLeftOfDecimal (n * Real.sqrt 2)

theorem a_is_irrational : Irrational a := by sorry

end a_is_irrational_l37_3720


namespace cup_volume_ratio_l37_3764

/-- Given a bottle that can be filled with 10 pours of cup a or 5 pours of cup b,
    prove that the volume of cup b is twice the volume of cup a. -/
theorem cup_volume_ratio (V A B : ℝ) (hA : 10 * A = V) (hB : 5 * B = V) :
  B = 2 * A := by sorry

end cup_volume_ratio_l37_3764


namespace nathan_family_storage_cost_l37_3782

/-- The cost to store items for a group at the temple shop -/
def storage_cost (num_people : ℕ) (objects_per_person : ℕ) (cost_per_object : ℕ) : ℕ :=
  num_people * objects_per_person * cost_per_object

/-- Proof that the storage cost for Nathan and his parents is 165 dollars -/
theorem nathan_family_storage_cost :
  storage_cost 3 5 11 = 165 := by
  sorry

end nathan_family_storage_cost_l37_3782


namespace polynomial_coefficient_sum_l37_3718

theorem polynomial_coefficient_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 - x^5 = a + a₁*(x + 4)^4*x + a₂*(x + 1)^3*x^2 + a₃*(x + 1)^2*x^3 + a₄*(x + 1)*x^4) →
  a₁ + a₃ = 15 := by
  sorry

end polynomial_coefficient_sum_l37_3718


namespace parabola_transformation_l37_3786

/-- Represents a parabola in the Cartesian plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

/-- The initial parabola y = -(x+1)^2 + 2 -/
def initial_parabola : Parabola :=
  { a := -1, h := 1, k := 2 }

/-- The final parabola y = -(x+2)^2 -/
def final_parabola : Parabola :=
  { a := -1, h := 2, k := 0 }

theorem parabola_transformation :
  (shift_vertical (shift_horizontal initial_parabola 1) (-2)) = final_parabola := by
  sorry

end parabola_transformation_l37_3786


namespace problem_solution_l37_3788

theorem problem_solution (p q : ℚ) 
  (h1 : 5 * p + 7 * q = 19)
  (h2 : 7 * p + 5 * q = 26) : 
  p = 29 / 8 := by
  sorry

end problem_solution_l37_3788


namespace exists_special_function_l37_3792

theorem exists_special_function :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f (n + 1)) = f (f n) + 2^(n - 1) := by
  sorry

end exists_special_function_l37_3792


namespace arithmetic_sequence_sum_l37_3727

theorem arithmetic_sequence_sum : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 28 →
  d = 3 →
  n * d = aₙ - a₁ + d →
  (n * (a₁ + aₙ)) / 2 = 145 :=
by
  sorry

end arithmetic_sequence_sum_l37_3727


namespace waffle_cooking_time_l37_3743

/-- The time it takes Carla to cook a batch of waffles -/
def waffle_time : ℕ := sorry

/-- The time it takes Carla to cook a chicken-fried steak -/
def steak_time : ℕ := 6

/-- The total time it takes Carla to cook 3 steaks and a batch of waffles -/
def total_time : ℕ := 28

/-- Theorem stating that the time to cook a batch of waffles is 10 minutes -/
theorem waffle_cooking_time : waffle_time = 10 := by sorry

end waffle_cooking_time_l37_3743


namespace three_glass_bottles_weight_l37_3715

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℝ := sorry

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℝ := sorry

/-- The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams -/
axiom total_weight : 4 * glass_bottle_weight + 5 * plastic_bottle_weight = 1050

/-- A glass bottle is 150 grams heavier than a plastic bottle -/
axiom weight_difference : glass_bottle_weight = plastic_bottle_weight + 150

/-- The weight of 3 glass bottles is 600 grams -/
theorem three_glass_bottles_weight : 3 * glass_bottle_weight = 600 := by sorry

end three_glass_bottles_weight_l37_3715


namespace jongkook_total_points_total_questions_sum_l37_3722

/-- The number of English problems solved by each student -/
def total_problems : ℕ := 18

/-- The number of 6-point questions Jongkook got correct -/
def correct_six_point : ℕ := 8

/-- The number of 5-point questions Jongkook got correct -/
def correct_five_point : ℕ := 6

/-- The point value of the first type of question -/
def points_type_one : ℕ := 6

/-- The point value of the second type of question -/
def points_type_two : ℕ := 5

/-- Theorem stating that Jongkook's total points is 78 -/
theorem jongkook_total_points :
  correct_six_point * points_type_one + correct_five_point * points_type_two = 78 := by
  sorry

/-- Theorem stating that the sum of correct questions equals the total number of problems -/
theorem total_questions_sum :
  correct_six_point + correct_five_point + (total_problems - correct_six_point - correct_five_point) = total_problems := by
  sorry

end jongkook_total_points_total_questions_sum_l37_3722


namespace paint_left_after_three_weeks_l37_3750

def paint_calculation (initial_paint : ℚ) : ℚ :=
  let after_week1 := initial_paint - (1/4 * initial_paint)
  let after_week2 := after_week1 - (1/2 * after_week1)
  let after_week3 := after_week2 - (2/3 * after_week2)
  after_week3

theorem paint_left_after_three_weeks :
  paint_calculation 360 = 45 := by sorry

end paint_left_after_three_weeks_l37_3750


namespace cubic_factorization_l37_3728

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end cubic_factorization_l37_3728


namespace original_number_is_84_l37_3765

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := n % 10 + n / 10

def swap_digits (n : ℕ) : ℕ := (n % 10) * 10 + n / 10

theorem original_number_is_84 (n : ℕ) 
  (h1 : is_two_digit n)
  (h2 : digit_sum n = 12)
  (h3 : n = swap_digits n + 36) :
  n = 84 := by
sorry

end original_number_is_84_l37_3765


namespace smallest_bob_number_l37_3772

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), 
    bob_number > 0 ∧
    has_all_prime_factors alice_number bob_number ∧
    (∀ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k → bob_number ≤ k) ∧
    bob_number = 6 := by
  sorry

end smallest_bob_number_l37_3772


namespace sqrt_5_simplest_l37_3770

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℝ), b ≠ 1 ∧ y = a * b^2

theorem sqrt_5_simplest :
  is_simplest_sqrt (Real.sqrt 5) ∧
  ¬is_simplest_sqrt (Real.sqrt 9) ∧
  ¬is_simplest_sqrt (Real.sqrt 18) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/2)) :=
sorry

end sqrt_5_simplest_l37_3770


namespace smallest_prime_factor_of_sum_l37_3714

theorem smallest_prime_factor_of_sum (n : ℕ) (m : ℕ) : 
  2 ∣ (2005^2007 + 2007^20015) ∧ 
  ∀ p : ℕ, p < 2 → p.Prime → ¬(p ∣ (2005^2007 + 2007^20015)) :=
by sorry

end smallest_prime_factor_of_sum_l37_3714


namespace quadratic_inequality_range_l37_3707

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 + (m + 1) * x + (m + 2) ≥ 0) ↔ m ≥ -1 := by
sorry

end quadratic_inequality_range_l37_3707


namespace root_equation_m_value_l37_3742

theorem root_equation_m_value (x m : ℝ) : 
  (3 / x = m / (x - 3)) → (x = 6) → (m = 3 / 2) := by
  sorry

end root_equation_m_value_l37_3742


namespace total_pupils_across_schools_l37_3740

theorem total_pupils_across_schools (
  girls_A boys_A girls_B boys_B girls_C boys_C : ℕ
) (h1 : girls_A = 542) (h2 : boys_A = 387)
  (h3 : girls_B = 713) (h4 : boys_B = 489)
  (h5 : girls_C = 628) (h6 : boys_C = 361) :
  girls_A + boys_A + girls_B + boys_B + girls_C + boys_C = 3120 := by
  sorry

end total_pupils_across_schools_l37_3740


namespace arithmetic_sequence_common_difference_l37_3717

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 7 - 2 * a 4 = 6) 
  (h3 : a 3 = 2) : 
  ∃ d : ℝ, (∀ n, a (n + 1) = a n + d) ∧ d = 4 := by
  sorry

end arithmetic_sequence_common_difference_l37_3717


namespace distribute_seven_balls_two_boxes_l37_3730

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 7 distinguishable balls into 2 distinguishable boxes is 128 -/
theorem distribute_seven_balls_two_boxes : 
  distribute_balls 7 2 = 128 := by
  sorry

end distribute_seven_balls_two_boxes_l37_3730


namespace largest_product_l37_3783

def digits : List Nat := [5, 6, 7, 8]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ digits ∧ (n % 10) ∈ digits

def valid_pair (a b : Nat) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (a / 10 ≠ b / 10) ∧ (a / 10 ≠ b % 10) ∧ (a % 10 ≠ b / 10) ∧ (a % 10 ≠ b % 10)

theorem largest_product :
  ∀ a b : Nat, valid_pair a b → a * b ≤ 3886 :=
by sorry

end largest_product_l37_3783


namespace cube_root_of_m_minus_n_l37_3713

theorem cube_root_of_m_minus_n (m n : ℝ) : 
  (3 * m + 2 * n = 36) → 
  (3 * n + 2 * m = 9) → 
  (m - n)^(1/3) = 3 := by
sorry

end cube_root_of_m_minus_n_l37_3713


namespace least_possible_average_of_four_integers_l37_3732

theorem least_possible_average_of_four_integers (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest integer is 90
  a ≥ 21 →                 -- Smallest integer is at least 21
  (a + b + c + d) / 4 ≥ 39 ∧ 
  ∃ (x y z w : ℤ), x < y ∧ y < z ∧ z < w ∧ w = 90 ∧ x ≥ 21 ∧ (x + y + z + w) / 4 = 39 :=
by
  sorry

#check least_possible_average_of_four_integers

end least_possible_average_of_four_integers_l37_3732


namespace pole_shortening_l37_3797

/-- Given a pole of length 20 meters that is shortened by 30%, prove that its new length is 14 meters. -/
theorem pole_shortening (original_length : ℝ) (shortening_percentage : ℝ) (new_length : ℝ) :
  original_length = 20 →
  shortening_percentage = 30 →
  new_length = original_length * (1 - shortening_percentage / 100) →
  new_length = 14 :=
by
  sorry

end pole_shortening_l37_3797


namespace reflection_of_S_l37_3729

-- Define the reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the composition of both reflections
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_neg_x (reflect_x_axis p)

-- Theorem statement
theorem reflection_of_S :
  double_reflection (5, 0) = (0, -5) := by
  sorry

end reflection_of_S_l37_3729


namespace long_division_puzzle_l37_3736

theorem long_division_puzzle : ∃! (a b c d : ℕ), 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧
  (c ≠ 0) ∧ (d ≠ 0) ∧
  (1000 * a + 100 * b + 10 * c + d) / (10 * c + d) = (100 * b + 10 * c + d) ∧
  (10 * c + d) * b = (10 * c + d) ∧
  (a = 3) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) := by
sorry

end long_division_puzzle_l37_3736


namespace tangent_point_exists_min_sum_of_squares_l37_3755

noncomputable section

-- Define the parabola C: x^2 = 2y
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the focus F(0, 1/2)
def focus : ℝ × ℝ := (0, 1/2)

-- Define the origin O(0, 0)
def origin : ℝ × ℝ := (0, 0)

-- Define a point M on the parabola in the first quadrant
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ M.1 > 0 ∧ M.2 > 0

-- Define the circle through M, F, and O with center Q
def circle_MFO (M Q : ℝ × ℝ) : Prop :=
  (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = (focus.1 - Q.1)^2 + (focus.2 - Q.2)^2 ∧
  (origin.1 - Q.1)^2 + (origin.2 - Q.2)^2 = (focus.1 - Q.1)^2 + (focus.2 - Q.2)^2

-- Distance from Q to the directrix is 3/4
def Q_to_directrix (Q : ℝ × ℝ) : Prop := Q.2 + 1/2 = 3/4

-- Theorem 1: Existence of point M where MQ is tangent to C
theorem tangent_point_exists :
  ∃ M : ℝ × ℝ, point_on_parabola M ∧
  ∃ Q : ℝ × ℝ, circle_MFO M Q ∧ Q_to_directrix Q ∧
  (M.1 = Real.sqrt 2 ∧ M.2 = 1) :=
sorry

-- Define the line l: y = kx + 1/4
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1/4

-- Theorem 2: Minimum value of |AB|^2 + |DE|^2
theorem min_sum_of_squares (k : ℝ) (h : 1/2 ≤ k ∧ k ≤ 2) :
  ∃ A B D E : ℝ × ℝ,
  point_on_parabola A ∧ point_on_parabola B ∧
  line k A.1 A.2 ∧ line k B.1 B.2 ∧
  (∃ Q : ℝ × ℝ, circle_MFO (Real.sqrt 2, 1) Q ∧ Q_to_directrix Q ∧
    line k D.1 D.2 ∧ line k E.1 E.2 ∧
    circle_MFO D Q ∧ circle_MFO E Q) ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (D.1 - E.1)^2 + (D.2 - E.2)^2 ≥ 13/2 :=
sorry

end tangent_point_exists_min_sum_of_squares_l37_3755


namespace equal_roots_iff_n_eq_neg_one_l37_3746

/-- The equation has equal roots if and only if n = -1 -/
theorem equal_roots_iff_n_eq_neg_one (n : ℝ) : 
  (∃! x : ℝ, x ≠ 2 ∧ (x * (x - 2) - (n + 2)) / ((x - 2) * (n - 2)) = x / n) ↔ n = -1 := by
  sorry

end equal_roots_iff_n_eq_neg_one_l37_3746


namespace solution_set_equivalence_l37_3741

theorem solution_set_equivalence :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end solution_set_equivalence_l37_3741


namespace big_dig_copper_production_l37_3758

/-- Represents a mine with its daily ore production and copper percentage -/
structure Mine where
  daily_production : ℝ
  copper_percentage : ℝ

/-- Calculates the total daily copper production from all mines -/
def total_copper_production (mines : List Mine) : ℝ :=
  mines.foldl (fun acc mine => acc + mine.daily_production * mine.copper_percentage) 0

/-- Theorem stating the total daily copper production from all four mines -/
theorem big_dig_copper_production :
  let mine_a : Mine := { daily_production := 4500, copper_percentage := 0.055 }
  let mine_b : Mine := { daily_production := 6000, copper_percentage := 0.071 }
  let mine_c : Mine := { daily_production := 5000, copper_percentage := 0.147 }
  let mine_d : Mine := { daily_production := 3500, copper_percentage := 0.092 }
  let all_mines : List Mine := [mine_a, mine_b, mine_c, mine_d]
  total_copper_production all_mines = 1730.5 := by
  sorry


end big_dig_copper_production_l37_3758


namespace intersection_point_l37_3753

/-- The line equation y = 5x - 6 -/
def line_equation (x y : ℝ) : Prop := y = 5 * x - 6

/-- The y-axis has the equation x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem intersection_point : 
  ∃ (x y : ℝ), line_equation x y ∧ y_axis x ∧ x = 0 ∧ y = -6 :=
sorry

end intersection_point_l37_3753


namespace equation_solution_l37_3704

theorem equation_solution :
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end equation_solution_l37_3704


namespace min_value_expression_l37_3775

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (9 * b) / (4 * a) + (a + b) / b ≥ 4 := by
  sorry

end min_value_expression_l37_3775


namespace not_all_greater_than_one_l37_3710

theorem not_all_greater_than_one (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) := by
  sorry

end not_all_greater_than_one_l37_3710


namespace max_surrounding_sum_l37_3759

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in a list are distinct -/
def all_distinct (l : List ℕ) : Prop := l.Nodup

/-- Checks if the product of three numbers equals 3240 -/
def product_is_3240 (a b c : ℕ) : Prop := a * b * c = 3240

/-- Checks if a grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  g 1 1 = 45 ∧
  (∀ i j k, (i = 0 ∧ j = k) ∨ (i = 2 ∧ j = k) ∨ (j = 0 ∧ i = k) ∨ (j = 2 ∧ i = k) ∨
            (i + j = 2 ∧ k = 1) ∨ (i = j ∧ k = 1) →
            product_is_3240 (g i j) (g i k) (g j k)) ∧
  all_distinct [g 0 0, g 0 1, g 0 2, g 1 0, g 1 2, g 2 0, g 2 1, g 2 2]

/-- Sum of the eight numbers surrounding the center in a grid -/
def surrounding_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 1 + g 0 2 + g 1 0 + g 1 2 + g 2 0 + g 2 1 + g 2 2

/-- The theorem stating the maximum sum of surrounding numbers -/
theorem max_surrounding_sum :
  ∀ g : Grid, valid_grid g → surrounding_sum g ≤ 160 :=
by sorry

end max_surrounding_sum_l37_3759


namespace negation_of_existence_negation_of_greater_than_3_l37_3745

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_greater_than_3 :
  (¬ ∃ x : ℝ, x^2 > 3) ↔ (∀ x : ℝ, x^2 ≤ 3) := by sorry

end negation_of_existence_negation_of_greater_than_3_l37_3745


namespace min_side_length_l37_3794

/-- Given two triangles PQR and SQR sharing side QR, prove that the minimum possible
    integral length of QR is 15 cm, given the lengths of other sides. -/
theorem min_side_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  ∃ (QR : ℕ), QR ≥ 15 ∧ ∀ (n : ℕ), n ≥ 15 → (n : ℝ) > PR - PQ ∧ (n : ℝ) > QS - SR :=
by sorry

end min_side_length_l37_3794


namespace diophantine_equation_solution_l37_3705

theorem diophantine_equation_solution : ∃ (a b c : ℕ+), a^3 + b^4 = c^5 ∧ a = 256 ∧ b = 64 ∧ c = 32 := by
  sorry

end diophantine_equation_solution_l37_3705


namespace problem_1_solution_problem_2_no_solution_l37_3789

-- Problem 1
theorem problem_1_solution (x : ℝ) :
  (x / (2*x - 5) + 5 / (5 - 2*x) = 1) ↔ (x = 0) :=
sorry

-- Problem 2
theorem problem_2_no_solution :
  ¬∃ (x : ℝ), ((2*x + 9) / (3*x - 9) = (4*x - 7) / (x - 3) + 2) :=
sorry

end problem_1_solution_problem_2_no_solution_l37_3789


namespace linear_function_parallel_and_point_l37_3757

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

-- Define parallel lines
def parallel (f g : ℝ → ℝ) : Prop := ∃ c : ℝ, ∀ x : ℝ, f x = g x + c

theorem linear_function_parallel_and_point :
  ∀ k b : ℝ,
  parallel (linear_function k b) (linear_function 2 1) →
  linear_function k b (-3) = 4 →
  linear_function k b = linear_function 2 10 :=
by sorry

end linear_function_parallel_and_point_l37_3757


namespace fraction_meaningful_l37_3716

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by
sorry

end fraction_meaningful_l37_3716


namespace difference_of_numbers_l37_3739

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 24) :
  |x - y| = 12/5 := by
  sorry

end difference_of_numbers_l37_3739


namespace solve_amusement_park_problem_l37_3799

def amusement_park_problem (adult_price child_price total_tickets child_tickets : ℕ) : Prop :=
  adult_price = 8 ∧
  child_price = 5 ∧
  total_tickets = 33 ∧
  child_tickets = 21 ∧
  (total_tickets - child_tickets) * adult_price + child_tickets * child_price = 201

theorem solve_amusement_park_problem :
  ∃ (adult_price child_price total_tickets child_tickets : ℕ),
    amusement_park_problem adult_price child_price total_tickets child_tickets :=
by
  sorry

end solve_amusement_park_problem_l37_3799


namespace function_passes_through_point_l37_3785

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ y : ℝ, y = a * 2 - 2 ∧ y = 1 := by
  sorry

end function_passes_through_point_l37_3785


namespace ab_range_l37_3752

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) :
  (a * b ≤ 1) ∨ (a * b ≥ 9) := by sorry

end ab_range_l37_3752


namespace arithmetic_expression_equality_l37_3780

theorem arithmetic_expression_equality : 8 + 18 / 3 - 4 * 2 = 6 := by
  sorry

end arithmetic_expression_equality_l37_3780


namespace coin_flip_theorem_l37_3778

/-- Represents the state of coins on a table -/
structure CoinState where
  total_coins : ℕ
  two_ruble_coins : ℕ
  five_ruble_coins : ℕ
  visible_sum : ℕ

/-- Checks if a CoinState is valid according to the problem conditions -/
def is_valid_state (state : CoinState) : Prop :=
  state.total_coins = 14 ∧
  state.two_ruble_coins + state.five_ruble_coins = state.total_coins ∧
  state.two_ruble_coins > 0 ∧
  state.five_ruble_coins > 0 ∧
  state.visible_sum ≤ 2 * state.two_ruble_coins + 5 * state.five_ruble_coins

/-- Calculates the new visible sum after flipping all coins -/
def flipped_sum (state : CoinState) : ℕ :=
  2 * state.two_ruble_coins + 5 * state.five_ruble_coins - state.visible_sum

/-- The main theorem to prove -/
theorem coin_flip_theorem (state : CoinState) :
  is_valid_state state →
  flipped_sum state = 3 * state.visible_sum →
  state.five_ruble_coins = 4 ∨ state.five_ruble_coins = 8 ∨ state.five_ruble_coins = 12 := by
  sorry


end coin_flip_theorem_l37_3778


namespace machine_value_depletion_rate_l37_3747

/-- The value depletion rate of a machine given its initial value and value after 2 years -/
theorem machine_value_depletion_rate 
  (initial_value : ℝ) 
  (value_after_two_years : ℝ) 
  (h1 : initial_value = 700) 
  (h2 : value_after_two_years = 567) : 
  ∃ (r : ℝ), 
    value_after_two_years = initial_value * (1 - r)^2 ∧ 
    r = 0.1 := by
  sorry

end machine_value_depletion_rate_l37_3747


namespace smallest_constant_inequality_l37_3734

theorem smallest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z + 2)) + Real.sqrt (y / (x + z + 2)) + Real.sqrt (z / (x + y + 2)) >
  (4 / Real.sqrt 3) * Real.cos (π / 6) := by
  sorry

end smallest_constant_inequality_l37_3734


namespace hyperbola_eccentricity_l37_3744

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is y = -√5/2 * x, then its eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5 / 2) : 
  Real.sqrt (a^2 + b^2) / a = 3/2 := by sorry

end hyperbola_eccentricity_l37_3744


namespace line_intersects_circle_l37_3760

/-- The line l: y = k(x - 1) intersects the circle C: x² + y² - 3x = 1 for any real number k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  y = k * (x - 1) ∧ x^2 + y^2 - 3*x = 1 := by
  sorry

end line_intersects_circle_l37_3760


namespace officer_hopps_ticket_problem_l37_3702

/-- Calculates the average number of tickets needed per day for the remaining days of the month -/
def average_tickets_remaining (total_tickets : ℕ) (days_in_month : ℕ) (first_period : ℕ) (first_period_average : ℕ) : ℚ :=
  let remaining_days := days_in_month - first_period
  let tickets_given := first_period * first_period_average
  let remaining_tickets := total_tickets - tickets_given
  (remaining_tickets : ℚ) / remaining_days

theorem officer_hopps_ticket_problem :
  average_tickets_remaining 200 31 15 8 = 5 := by
  sorry

end officer_hopps_ticket_problem_l37_3702


namespace sum_not_divisible_l37_3798

theorem sum_not_divisible : ∃ (y : ℤ), 
  y = 42 + 98 + 210 + 333 + 175 + 28 ∧ 
  ¬(∃ (k : ℤ), y = 7 * k) ∧ 
  ¬(∃ (k : ℤ), y = 14 * k) ∧ 
  ¬(∃ (k : ℤ), y = 28 * k) ∧ 
  ¬(∃ (k : ℤ), y = 21 * k) := by
  sorry

end sum_not_divisible_l37_3798


namespace area_ratio_equals_side_ratio_l37_3751

/-- Triangle PQR with angle bisector PS -/
structure AngleBisectorTriangle where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- PS is an angle bisector -/
  PS_is_angle_bisector : Bool

/-- The ratio of areas of triangles formed by an angle bisector -/
def area_ratio (t : AngleBisectorTriangle) : ℝ :=
  sorry

/-- Theorem: The ratio of areas of triangles formed by an angle bisector
    is equal to the ratio of the lengths of the sides adjacent to the bisected angle -/
theorem area_ratio_equals_side_ratio (t : AngleBisectorTriangle) 
  (h : t.PS_is_angle_bisector = true) (h1 : t.PQ = 45) (h2 : t.PR = 75) (h3 : t.QR = 64) : 
  area_ratio t = 3 / 5 := by
  sorry

end area_ratio_equals_side_ratio_l37_3751


namespace sin_cos_shift_l37_3731

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.cos (2 * (x + π / 12) - π / 3) := by
  sorry

end sin_cos_shift_l37_3731


namespace batsman_average_after_12th_innings_average_increase_by_2_min_score_before_12th_consecutive_scores_before_12th_l37_3768

/-- Represents a cricket batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  minScore : Nat
  consecutiveScores : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : Rat :=
  b.totalRuns / b.innings

/-- Represents the batsman's performance after 11 innings -/
def initialBatsman : Batsman :=
  { innings := 11
  , totalRuns := 11 * 24  -- 11 * average before 12th innings
  , minScore := 20
  , consecutiveScores := 25 }

/-- Represents the batsman's performance after 12 innings -/
def finalBatsman : Batsman :=
  { innings := 12
  , totalRuns := initialBatsman.totalRuns + 48
  , minScore := 20
  , consecutiveScores := 25 }

theorem batsman_average_after_12th_innings :
  average finalBatsman = 26 := by
  sorry

theorem average_increase_by_2 :
  average finalBatsman - average initialBatsman = 2 := by
  sorry

theorem min_score_before_12th :
  initialBatsman.minScore ≥ 20 := by
  sorry

theorem consecutive_scores_before_12th :
  initialBatsman.consecutiveScores = 25 := by
  sorry

end batsman_average_after_12th_innings_average_increase_by_2_min_score_before_12th_consecutive_scores_before_12th_l37_3768


namespace negation_of_all_squares_positive_l37_3706

theorem negation_of_all_squares_positive :
  (¬ ∀ n : ℕ, n^2 > 0) ↔ (∃ n : ℕ, ¬(n^2 > 0)) :=
by sorry

end negation_of_all_squares_positive_l37_3706


namespace price_reduction_l37_3766

theorem price_reduction (initial_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  first_reduction = 0.15 →
  second_reduction = 0.20 →
  let price_after_first := initial_price * (1 - first_reduction)
  let price_after_second := price_after_first * (1 - second_reduction)
  initial_price > 0 →
  (initial_price - price_after_second) / initial_price = 0.32 := by
  sorry

end price_reduction_l37_3766


namespace total_oranges_l37_3737

theorem total_oranges (children : ℕ) (oranges_per_child : ℕ) 
  (h1 : children = 4) 
  (h2 : oranges_per_child = 3) : 
  children * oranges_per_child = 12 := by
  sorry

end total_oranges_l37_3737


namespace odd_integer_divides_power_factorial_minus_one_l37_3776

theorem odd_integer_divides_power_factorial_minus_one (n : ℕ) (h_odd : Odd n) (h_ge_one : n ≥ 1) :
  n ∣ 2^(n!) - 1 := by
  sorry

end odd_integer_divides_power_factorial_minus_one_l37_3776


namespace range_of_a_l37_3738

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^2 + (1-a)*x + 3-a > 0) ↔ a < 3 :=
by sorry

end range_of_a_l37_3738


namespace instructors_reunion_l37_3733

/-- The number of weeks between Rita's teaching sessions -/
def rita_weeks : ℕ := 5

/-- The number of weeks between Pedro's teaching sessions -/
def pedro_weeks : ℕ := 8

/-- The number of weeks between Elaine's teaching sessions -/
def elaine_weeks : ℕ := 10

/-- The number of weeks between Moe's teaching sessions -/
def moe_weeks : ℕ := 9

/-- The number of weeks until all instructors teach together again -/
def weeks_until_reunion : ℕ := 360

theorem instructors_reunion :
  Nat.lcm rita_weeks (Nat.lcm pedro_weeks (Nat.lcm elaine_weeks moe_weeks)) = weeks_until_reunion :=
sorry

end instructors_reunion_l37_3733


namespace theresa_has_eleven_games_l37_3712

/-- The number of video games Tory has -/
def tory_games : ℕ := 6

/-- The number of video games Julia has -/
def julia_games : ℕ := tory_games / 3

/-- The number of video games Theresa has -/
def theresa_games : ℕ := 3 * julia_games + 5

/-- Theorem stating that Theresa has 11 video games -/
theorem theresa_has_eleven_games : theresa_games = 11 := by
  sorry

end theresa_has_eleven_games_l37_3712


namespace project_cost_sharing_l37_3791

/-- Given initial payments P and Q, and an additional cost R, 
    calculate the amount Javier must pay to Cora for equal cost sharing. -/
theorem project_cost_sharing 
  (P Q R : ℝ) 
  (h1 : R = 3 * Q - 2 * P) 
  (h2 : P < Q) : 
  (2 * Q - P) / 2 = (P + Q + R) / 2 - Q := by sorry

end project_cost_sharing_l37_3791


namespace toby_money_left_l37_3735

/-- The amount of money Toby received -/
def total_amount : ℚ := 343

/-- The number of brothers Toby has -/
def num_brothers : ℕ := 2

/-- The number of cousins Toby has -/
def num_cousins : ℕ := 4

/-- The percentage of money each brother receives -/
def brother_percentage : ℚ := 12 / 100

/-- The percentage of money each cousin receives -/
def cousin_percentage : ℚ := 7 / 100

/-- The percentage of money spent on mom's gift -/
def mom_gift_percentage : ℚ := 15 / 100

/-- The amount left for Toby after sharing and buying the gift -/
def amount_left : ℚ := 
  total_amount - 
  (num_brothers * (brother_percentage * total_amount) + 
   num_cousins * (cousin_percentage * total_amount) + 
   mom_gift_percentage * total_amount)

theorem toby_money_left : amount_left = 113.19 := by
  sorry

end toby_money_left_l37_3735


namespace closest_integer_to_sqrt17_minus_1_l37_3774

theorem closest_integer_to_sqrt17_minus_1 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (Real.sqrt 17 - 1)| ≤ |m - (Real.sqrt 17 - 1)| ∧ n = 3 := by
  sorry

end closest_integer_to_sqrt17_minus_1_l37_3774


namespace orange_groups_count_l37_3796

/-- The number of groups of oranges in Philip's collection -/
def orange_groups (total_oranges : ℕ) (oranges_per_group : ℕ) : ℕ :=
  total_oranges / oranges_per_group

/-- Theorem stating that the number of orange groups is 16 -/
theorem orange_groups_count :
  orange_groups 384 24 = 16 := by
  sorry

end orange_groups_count_l37_3796


namespace sam_pennies_total_l37_3711

/-- Given that Sam had 98 pennies initially and found 93 more pennies,
    prove that he now has 191 pennies in total. -/
theorem sam_pennies_total (initial : ℕ) (found : ℕ) (h1 : initial = 98) (h2 : found = 93) :
  initial + found = 191 := by
  sorry

end sam_pennies_total_l37_3711


namespace book_distribution_l37_3756

/-- The number of ways to distribute n distinct books among k people, 
    with each person receiving m books -/
def distribute_books (n k m : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n distinct items -/
def choose (n r : ℕ) : ℕ := sorry

theorem book_distribution :
  distribute_books 6 3 2 = 90 :=
by
  sorry

end book_distribution_l37_3756


namespace a_alone_time_l37_3701

-- Define the work rates of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
axiom a_twice_b : a = 2 * b
axiom c_half_b : c = 0.5 * b
axiom combined_rate : a + b + c = 1 / 18
axiom c_alone_rate : c = 1 / 36

-- Theorem to prove
theorem a_alone_time : (1 / a) = 31.5 := by sorry

end a_alone_time_l37_3701


namespace twelfth_term_value_l37_3795

/-- The nth term of a geometric sequence -/
def geometric_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

/-- The 12th term of the specific geometric sequence -/
def twelfth_term : ℚ :=
  geometric_term 5 (2/5) 12

theorem twelfth_term_value : twelfth_term = 10240/48828125 := by
  sorry

end twelfth_term_value_l37_3795


namespace election_ratio_l37_3748

theorem election_ratio :
  ∀ (R D : ℝ),
  R > 0 → D > 0 →
  (0.70 * R + 0.25 * D) - (0.30 * R + 0.75 * D) = 0.039999999999999853 * (R + D) →
  R / D = 1.5 := by
sorry

end election_ratio_l37_3748


namespace no_given_factors_of_polynomial_l37_3779

theorem no_given_factors_of_polynomial :
  let p (x : ℝ) := x^4 - 2*x^2 + 9
  let factors := [
    (fun x => x^2 + 3),
    (fun x => x + 1),
    (fun x => x^2 - 3),
    (fun x => x^2 + 2*x - 3)
  ]
  ∀ f ∈ factors, ¬ (∃ q : ℝ → ℝ, ∀ x, p x = f x * q x) :=
by sorry

end no_given_factors_of_polynomial_l37_3779


namespace valve_emission_difference_l37_3787

/-- The difference in water emission rates between two valves filling a pool -/
theorem valve_emission_difference (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) : 
  pool_capacity > 0 → 
  both_valves_time > 0 → 
  first_valve_time > 0 → 
  pool_capacity / both_valves_time - pool_capacity / first_valve_time = 50 := by
  sorry

#check valve_emission_difference 12000 48 120

end valve_emission_difference_l37_3787


namespace ribbon_distribution_l37_3719

theorem ribbon_distribution (total_ribbon : ℚ) (num_boxes : ℕ) :
  total_ribbon = 2 / 5 →
  num_boxes = 5 →
  (total_ribbon / num_boxes : ℚ) = 2 / 25 := by
sorry

end ribbon_distribution_l37_3719


namespace abs_sum_minimum_l37_3777

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end abs_sum_minimum_l37_3777


namespace vertex_on_x_axis_l37_3700

/-- The parabola equation -/
def parabola (x c : ℝ) : ℝ := x^2 - 8*x + c

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 4

/-- Theorem: The vertex of the parabola y = x^2 - 8x + c lies on the x-axis if and only if c = 16 -/
theorem vertex_on_x_axis (c : ℝ) : 
  parabola vertex_x c = 0 ↔ c = 16 := by
sorry

end vertex_on_x_axis_l37_3700


namespace inequality_proof_l37_3725

theorem inequality_proof (x : ℝ) : 
  2 < x → x < 9/2 → (10*x^2 + 15*x - 75) / ((3*x - 6)*(x + 5)) < 4 := by
  sorry

end inequality_proof_l37_3725


namespace arctan_difference_of_tans_l37_3703

theorem arctan_difference_of_tans : 
  let result := Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (20 * π / 180))
  0 ≤ result ∧ result ≤ π ∧ result = 15 * π / 180 := by
  sorry

end arctan_difference_of_tans_l37_3703


namespace cubic_equation_solution_l37_3761

theorem cubic_equation_solution (x y : ℝ) (h1 : x^(3*y) = 27) (h2 : x = 3) : y = 1 := by
  sorry

end cubic_equation_solution_l37_3761


namespace parallelogram_altitude_theorem_l37_3708

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)

-- Define the properties of the parallelogram
def Parallelogram.DC (p : Parallelogram) : ℝ := sorry
def Parallelogram.EB (p : Parallelogram) : ℝ := sorry
def Parallelogram.DE (p : Parallelogram) : ℝ := sorry
def Parallelogram.DF (p : Parallelogram) : ℝ := sorry

-- State the theorem
theorem parallelogram_altitude_theorem (p : Parallelogram) 
  (h1 : p.DC = 15)
  (h2 : p.EB = 5)
  (h3 : p.DE = 9) :
  p.DF = 9 := by sorry

end parallelogram_altitude_theorem_l37_3708


namespace lunch_breakfast_difference_l37_3784

def breakfast_cost : ℚ := 2 + 3 + 4 + 3.5 + 1.5

def lunch_base_cost : ℚ := 3.5 + 4 + 5.25 + 6 + 1 + 3

def service_charge (cost : ℚ) : ℚ := cost * (1 + 0.1)

def food_tax (cost : ℚ) : ℚ := cost * (1 + 0.05)

def lunch_total_cost : ℚ := food_tax (service_charge lunch_base_cost)

theorem lunch_breakfast_difference :
  lunch_total_cost - breakfast_cost = 12.28 := by sorry

end lunch_breakfast_difference_l37_3784


namespace orange_count_l37_3781

theorem orange_count (total : ℕ) (apple_ratio : ℕ) (orange_count : ℕ) : 
  total = 40 →
  apple_ratio = 3 →
  orange_count + apple_ratio * orange_count = total →
  orange_count = 10 := by
sorry

end orange_count_l37_3781


namespace discounted_price_calculation_l37_3773

/-- The actual price of the good in Rupees -/
def actual_price : ℝ := 9502.923976608186

/-- The first discount rate -/
def discount1 : ℝ := 0.20

/-- The second discount rate -/
def discount2 : ℝ := 0.10

/-- The third discount rate -/
def discount3 : ℝ := 0.05

/-- The discounted price after applying three successive discounts -/
def discounted_price (p : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  p * (1 - d1) * (1 - d2) * (1 - d3)

/-- Theorem stating that the discounted price is approximately 6498.40 -/
theorem discounted_price_calculation :
  ∃ ε > 0, abs (discounted_price actual_price discount1 discount2 discount3 - 6498.40) < ε :=
sorry

end discounted_price_calculation_l37_3773


namespace monotonic_increasing_interval_l37_3754

noncomputable def f (a x : ℝ) : ℝ := a^(x^2 - 3*x + 2)

theorem monotonic_increasing_interval 
  (a : ℝ) 
  (h : a > 1) :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 3/2 ∧ x₂ ≥ 3/2 ∧ x₁ < x₂ → f a x₁ < f a x₂ :=
by sorry

end monotonic_increasing_interval_l37_3754


namespace not_p_sufficient_not_necessary_for_not_q_l37_3790

theorem not_p_sufficient_not_necessary_for_not_q :
  ∃ (x : ℝ), (¬(|x + 1| > 2) → ¬(5*x - 6 > x^2)) ∧
             ∃ (y : ℝ), ¬(5*y - 6 > y^2) ∧ (|y + 1| > 2) := by
  sorry

end not_p_sufficient_not_necessary_for_not_q_l37_3790


namespace parabola_equation_l37_3724

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the left vertex of the hyperbola
def left_vertex : ℝ × ℝ := (-3, 0)

-- Define the point P
def point_P : ℝ × ℝ := (2, -4)

-- Define the parabola equations
def parabola1 (x y : ℝ) : Prop := y^2 = 8 * x
def parabola2 (x y : ℝ) : Prop := x^2 = -y

-- Theorem statement
theorem parabola_equation :
  ∀ (f : ℝ × ℝ) (p : (ℝ × ℝ) → Prop),
    (f = left_vertex) →  -- The focus of the parabola is the left vertex of the hyperbola
    (p point_P) →  -- The parabola passes through point P
    (∀ (x y : ℝ), p (x, y) ↔ (parabola1 x y ∨ parabola2 x y)) :=
by sorry

end parabola_equation_l37_3724


namespace rectangle_circle_area_ratio_l37_3793

theorem rectangle_circle_area_ratio :
  ∀ (w r : ℝ),
  w > 0 → r > 0 →
  6 * w = 2 * π * r →
  (2 * w * w) / (π * r^2) = 2 * π / 9 := by
sorry

end rectangle_circle_area_ratio_l37_3793


namespace volleyball_club_girls_l37_3771

theorem volleyball_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 36 →
  present = 24 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = present →
  girls = 18 :=
by sorry

end volleyball_club_girls_l37_3771
