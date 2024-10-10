import Mathlib

namespace sum_minimized_at_6_l1019_101911

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = -11
  sum_of_4th_and_6th : a 4 + a 6 = -6

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The value of n that minimizes the sum of first n terms -/
def minimizing_n (seq : ArithmeticSequence) : ℕ :=
  6

theorem sum_minimized_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n_terms seq (minimizing_n seq) ≤ sum_n_terms seq n :=
sorry

end sum_minimized_at_6_l1019_101911


namespace paper_I_passing_percentage_l1019_101972

/-- Calculates the passing percentage for an exam given the maximum marks,
    the marks secured by a candidate, and the marks by which they failed. -/
def calculate_passing_percentage (max_marks : ℕ) (secured_marks : ℕ) (failed_by : ℕ) : ℚ :=
  let passing_marks : ℕ := secured_marks + failed_by
  (passing_marks : ℚ) / max_marks * 100

/-- Theorem stating that the passing percentage for Paper I is 40% -/
theorem paper_I_passing_percentage :
  calculate_passing_percentage 150 40 20 = 40 := by
sorry

end paper_I_passing_percentage_l1019_101972


namespace complement_of_A_in_U_l1019_101966

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 3, 5, 7}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4, 6} := by sorry

end complement_of_A_in_U_l1019_101966


namespace binomial_coefficient_26_6_l1019_101943

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end binomial_coefficient_26_6_l1019_101943


namespace arithmetic_seq_sum_l1019_101914

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 3) 
  (h6 : seq.S 6 = 15) : 
  seq.a 10 + seq.a 11 + seq.a 12 = 30 := by
  sorry


end arithmetic_seq_sum_l1019_101914


namespace triangle_area_l1019_101948

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → a + c > b → 
  (a : ℝ) * (b : ℝ) / 2 = 6 :=
sorry

end triangle_area_l1019_101948


namespace fraction_used_is_47_48_l1019_101967

/-- Represents the car's journey with given parameters -/
structure CarJourney where
  tankCapacity : ℚ
  firstLegDuration : ℚ
  firstLegSpeed : ℚ
  firstLegConsumptionRate : ℚ
  refillAmount : ℚ
  secondLegDuration : ℚ
  secondLegSpeed : ℚ
  secondLegConsumptionRate : ℚ

/-- Calculates the fraction of a full tank used after the entire journey -/
def fractionUsed (journey : CarJourney) : ℚ :=
  let firstLegDistance := journey.firstLegDuration * journey.firstLegSpeed
  let firstLegUsed := firstLegDistance / journey.firstLegConsumptionRate
  let secondLegDistance := journey.secondLegDuration * journey.secondLegSpeed
  let secondLegUsed := secondLegDistance / journey.secondLegConsumptionRate
  (firstLegUsed + secondLegUsed) / journey.tankCapacity

/-- The specific journey described in the problem -/
def specificJourney : CarJourney :=
  { tankCapacity := 12
  , firstLegDuration := 3
  , firstLegSpeed := 50
  , firstLegConsumptionRate := 40
  , refillAmount := 5
  , secondLegDuration := 4
  , secondLegSpeed := 60
  , secondLegConsumptionRate := 30
  }

/-- Theorem stating that the fraction of tank used in the specific journey is 47/48 -/
theorem fraction_used_is_47_48 : fractionUsed specificJourney = 47 / 48 := by
  sorry


end fraction_used_is_47_48_l1019_101967


namespace parabola_point_value_l1019_101988

/-- Given points A(a,m), B(b,m), P(a+b,n) on the parabola y=x^2-2x-2, prove that n = -2 -/
theorem parabola_point_value (a b m n : ℝ) : 
  (m = a^2 - 2*a - 2) →  -- A is on the parabola
  (m = b^2 - 2*b - 2) →  -- B is on the parabola
  (n = (a+b)^2 - 2*(a+b) - 2) →  -- P is on the parabola
  (n = -2) := by
sorry

end parabola_point_value_l1019_101988


namespace club_officer_selection_l1019_101913

theorem club_officer_selection (n : ℕ) (e : ℕ) (h1 : n = 12) (h2 : e = 5) (h3 : e ≤ n) :
  (n * (n - 1) * e * (n - 2)) = 6600 :=
sorry

end club_officer_selection_l1019_101913


namespace arithmetic_sequence_fifth_term_l1019_101942

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 6) : 
  a 5 = 2 := by
sorry

end arithmetic_sequence_fifth_term_l1019_101942


namespace log_properties_l1019_101958

-- Define the logarithm function for base b
noncomputable def log_b (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_properties (b : ℝ) (h : 0 < b ∧ b < 1) :
  (log_b b 1 = 0) ∧ 
  (log_b b b = 1) ∧ 
  (∀ x : ℝ, 1 < x → x < b → log_b b x > 0) ∧
  (∀ x y : ℝ, 1 < x → x < y → y < b → log_b b x > log_b b y) :=
by sorry

end log_properties_l1019_101958


namespace area_of_triangle_ABC_l1019_101979

-- Define the points in the plane
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AC : ℝ := 15
def AB : ℝ := 17
def DC : ℝ := 9

-- Define the angle D as a right angle
def angle_D_is_right : Prop := sorry

-- Define that the points are coplanar
def points_are_coplanar : Prop := sorry

-- Define the area of triangle ABC
def area_ABC : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC :
  points_are_coplanar →
  angle_D_is_right →
  area_ABC = 54 + 6 * Real.sqrt 145 :=
sorry

end area_of_triangle_ABC_l1019_101979


namespace max_perimeter_right_triangle_l1019_101982

theorem max_perimeter_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 36) : 
  a + b + 6 ≤ 6 * Real.sqrt 2 + 6 :=
sorry

end max_perimeter_right_triangle_l1019_101982


namespace geometric_sequence_property_l1019_101983

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) 
  (h_sum : a 2013 + a 2015 = ∫ x in (0:ℝ)..2, Real.sqrt (4 - x^2)) :
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 := by
  sorry

end geometric_sequence_property_l1019_101983


namespace cubic_roots_relation_l1019_101903

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x + 10 = (x - p) * (x - q) * (x - r)) →
  (∀ x : ℝ, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 80 := by
sorry

end cubic_roots_relation_l1019_101903


namespace cory_fruit_orders_l1019_101952

def number_of_orders (apples oranges lemons : ℕ) : ℕ :=
  Nat.factorial (apples + oranges + lemons) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial lemons)

theorem cory_fruit_orders :
  number_of_orders 4 2 1 = 105 := by
  sorry

end cory_fruit_orders_l1019_101952


namespace ginos_popsicle_sticks_l1019_101905

def my_popsicle_sticks : ℕ := 50
def total_popsicle_sticks : ℕ := 113

theorem ginos_popsicle_sticks :
  total_popsicle_sticks - my_popsicle_sticks = 63 := by sorry

end ginos_popsicle_sticks_l1019_101905


namespace tangent_line_to_circle_l1019_101907

theorem tangent_line_to_circle (p : ℝ) : 
  (∀ x y : ℝ, x = -p/2 → x^2 + y^2 + 6*x + 8 = 0 → 
    ∃! y : ℝ, x^2 + y^2 + 6*x + 8 = 0) → 
  p = 4 ∨ p = 8 := by
sorry

end tangent_line_to_circle_l1019_101907


namespace dance_off_combined_time_l1019_101904

/-- Given John and James' dancing schedules, prove their combined dancing time is 20 hours --/
theorem dance_off_combined_time (john_first_session : ℝ) (john_break : ℝ) (john_second_session : ℝ) 
  (james_extra_fraction : ℝ) : 
  john_first_session = 3 ∧ 
  john_break = 1 ∧ 
  john_second_session = 5 ∧ 
  james_extra_fraction = 1/3 → 
  (john_first_session + john_second_session) + 
  ((john_first_session + john_break + john_second_session) + 
   (john_first_session + john_break + john_second_session) * james_extra_fraction) = 20 := by
sorry

end dance_off_combined_time_l1019_101904


namespace parabola_parameters_correct_l1019_101919

/-- Two parabolas with common focus and passing through two points -/
structure TwoParabolas where
  F : ℝ × ℝ
  P₁ : ℝ × ℝ
  P₂ : ℝ × ℝ
  h₁ : F = (2, 2)
  h₂ : P₁ = (4, 2)
  h₃ : P₂ = (-2, 5)

/-- The parameters of the two parabolas -/
def parabola_parameters (tp : TwoParabolas) : ℝ × ℝ :=
  (2, 3.6)

/-- Theorem stating that the parameters of the two parabolas are 2 and 3.6 -/
theorem parabola_parameters_correct (tp : TwoParabolas) :
  parabola_parameters tp = (2, 3.6) := by
  sorry

end parabola_parameters_correct_l1019_101919


namespace quadratic_one_solution_l1019_101945

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + k = 0) ↔ k = 49/12 := by
sorry

end quadratic_one_solution_l1019_101945


namespace school_teachers_count_l1019_101957

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sample_students : ℕ) : 
  total = 2400 →
  sample_size = 320 →
  sample_students = 280 →
  ∃ (teachers students : ℕ),
    teachers + students = total ∧
    teachers * sample_students = students * (sample_size - sample_students) ∧
    teachers = 300 := by
  sorry

end school_teachers_count_l1019_101957


namespace three_digit_number_sum_l1019_101929

theorem three_digit_number_sum (a b c : ℕ) : 
  (100 * a + 10 * b + c) % 5 = 0 →
  a = 2 * b →
  a * b * c = 40 →
  a + b + c = 11 :=
by sorry

end three_digit_number_sum_l1019_101929


namespace paul_shopping_money_left_l1019_101980

theorem paul_shopping_money_left 
  (initial_money : ℝ)
  (bread_price : ℝ)
  (butter_original_price : ℝ)
  (butter_discount : ℝ)
  (juice_price_multiplier : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : initial_money = 15)
  (h2 : bread_price = 2)
  (h3 : butter_original_price = 3)
  (h4 : butter_discount = 0.1)
  (h5 : juice_price_multiplier = 2)
  (h6 : sales_tax_rate = 0.05) :
  initial_money - 
  ((bread_price + 
    (butter_original_price * (1 - butter_discount)) + 
    (bread_price * juice_price_multiplier)) * 
   (1 + sales_tax_rate)) = 5.86 := by
sorry

end paul_shopping_money_left_l1019_101980


namespace ball_distribution_count_l1019_101925

/-- Represents a valid distribution of balls into boxes -/
structure BallDistribution where
  x : ℕ
  y : ℕ
  z : ℕ
  sum_eq_7 : x + y + z = 7
  ordered : x ≥ y ∧ y ≥ z

/-- The number of ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes -/
def distributionCount : ℕ := sorry

theorem ball_distribution_count : distributionCount = 8 := by sorry

end ball_distribution_count_l1019_101925


namespace sum_of_squares_l1019_101993

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
sorry

end sum_of_squares_l1019_101993


namespace sum_of_inscribed_circles_limit_l1019_101938

/-- The sum of areas of inscribed circles in a rectangle --/
def sum_of_circle_areas (m : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The limit of the sum as n approaches infinity --/
def limit_of_sum (m : ℝ) : ℝ :=
  sorry

/-- Theorem: The limit of the sum of areas of inscribed circles approaches 5πm^2 --/
theorem sum_of_inscribed_circles_limit (m : ℝ) (h : m > 0) :
  limit_of_sum m = 5 * Real.pi * m^2 := by
  sorry

end sum_of_inscribed_circles_limit_l1019_101938


namespace equation_solution_l1019_101912

theorem equation_solution :
  ∃! x : ℤ, 45 - (28 - (x - (15 - 17))) = 56 :=
by
  -- The unique solution is x = 19
  use 19
  constructor
  · -- Prove that x = 19 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end equation_solution_l1019_101912


namespace unique_modular_congruence_l1019_101991

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ -250 ≡ n [ZMOD 17] ∧ n = 5 := by
  sorry

end unique_modular_congruence_l1019_101991


namespace cream_cheese_amount_l1019_101973

/-- Calculates the amount of cream cheese used in a spinach quiche recipe. -/
theorem cream_cheese_amount
  (raw_spinach : ℝ)
  (cooked_spinach_percentage : ℝ)
  (eggs : ℝ)
  (total_volume : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_percentage = 0.20)
  (h3 : eggs = 4)
  (h4 : total_volume = 18) :
  total_volume - (raw_spinach * cooked_spinach_percentage) - eggs = 6 := by
  sorry

end cream_cheese_amount_l1019_101973


namespace quadratic_root_property_l1019_101934

theorem quadratic_root_property (m : ℝ) : 
  m^2 - m - 3 = 0 → 2023 - m^2 + m = 2020 := by
  sorry

end quadratic_root_property_l1019_101934


namespace parlor_game_solution_l1019_101990

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : a > 0

/-- Calculates the sum of permutations for a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.c + n.b +
  100 * n.a + 10 * n.b + n.c +
  100 * n.b + 10 * n.c + n.a +
  100 * n.b + 10 * n.a + n.c +
  100 * n.c + 10 * n.a + n.b +
  100 * n.c + 10 * n.b + n.a

/-- The main theorem -/
theorem parlor_game_solution :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = 4326 ∧ n.a = 3 ∧ n.b = 9 ∧ n.c = 0 := by
  sorry

end parlor_game_solution_l1019_101990


namespace cubic_fraction_equality_l1019_101922

theorem cubic_fraction_equality : 
  let a : ℚ := 7
  let b : ℚ := 6
  let c : ℚ := 1
  (a^3 + b^3) / (a^2 - a*b + b^2 + c) = 559 / 44 := by sorry

end cubic_fraction_equality_l1019_101922


namespace cat_arrangements_l1019_101921

def number_of_arrangements (n : ℕ) : ℕ := Nat.factorial n

theorem cat_arrangements : number_of_arrangements 3 = 6 := by sorry

end cat_arrangements_l1019_101921


namespace shaded_area_calculation_l1019_101984

theorem shaded_area_calculation (carpet_side : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) :
  carpet_side = 12 →
  carpet_side / large_square_side = 2 →
  large_square_side / small_square_side = 2 →
  12 * (small_square_side ^ 2) + large_square_side ^ 2 = 144 := by
  sorry

#check shaded_area_calculation

end shaded_area_calculation_l1019_101984


namespace rectangle_area_problem_l1019_101923

theorem rectangle_area_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 3) * (y - 1) = x * y ∧
  (x - 3) * (y + 1.5) = x * y ∧
  x * y = 90 := by
  sorry

end rectangle_area_problem_l1019_101923


namespace election_result_l1019_101975

/-- Represents an election with three candidates -/
structure Election :=
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)

/-- The election satisfies the given conditions -/
def valid_election (e : Election) : Prop :=
  e.votes_A = (35 * e.total_votes) / 100 ∧
  e.votes_C = (25 * e.total_votes) / 100 ∧
  e.votes_B = e.votes_A + 2460 ∧
  e.total_votes = e.votes_A + e.votes_B + e.votes_C

theorem election_result (e : Election) (h : valid_election e) :
  e.votes_B = (40 * e.total_votes) / 100 ∧ e.total_votes = 49200 := by
  sorry


end election_result_l1019_101975


namespace second_tree_groups_count_l1019_101963

/-- Represents the number of rings in a group -/
def rings_per_group : ℕ := 6

/-- Represents the number of ring groups in the first tree -/
def first_tree_groups : ℕ := 70

/-- Represents the age difference between the first and second tree in years -/
def age_difference : ℕ := 180

/-- Calculates the number of ring groups in the second tree -/
def second_tree_groups : ℕ := 
  (first_tree_groups * rings_per_group - age_difference) / rings_per_group

theorem second_tree_groups_count : second_tree_groups = 40 := by
  sorry

end second_tree_groups_count_l1019_101963


namespace add_point_four_five_to_fifty_seven_point_two_five_l1019_101928

theorem add_point_four_five_to_fifty_seven_point_two_five :
  57.25 + 0.45 = 57.7 := by
  sorry

end add_point_four_five_to_fifty_seven_point_two_five_l1019_101928


namespace problem_solution_l1019_101968

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b * log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - x

theorem problem_solution :
  (∀ x > 0, Monotone (F (1/8))) ∧
  (∀ a ≥ 1/8, ∀ x > 0, Monotone (F a)) ∧
  (∃ b : ℝ, (b < -2 ∨ b > (ℯ^2 + 2)/(ℯ - 1)) ↔
    ∃ x₀ ∈ Set.Icc 1 ℯ, x₀ - f b x₀ < -(1 + b)/x₀) := by sorry

end problem_solution_l1019_101968


namespace melanie_dimes_given_l1019_101976

/-- The number of dimes Melanie gave to her dad -/
def dimes_given_to_dad : ℕ := 7

/-- The initial number of dimes Melanie had -/
def initial_dimes : ℕ := 8

/-- The number of dimes Melanie received from her mother -/
def dimes_from_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 5

theorem melanie_dimes_given :
  initial_dimes - dimes_given_to_dad + dimes_from_mother = current_dimes :=
by sorry

end melanie_dimes_given_l1019_101976


namespace find_M_l1019_101969

theorem find_M : ∃ M : ℕ+, (15^2 * 25^2 : ℕ) = 5^2 * M^2 ∧ M = 375 := by
  sorry

end find_M_l1019_101969


namespace green_ball_probability_l1019_101944

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containerX : Container := ⟨5, 5⟩
def containerY : Container := ⟨7, 3⟩
def containerZ : Container := ⟨7, 3⟩

/-- The list of all containers -/
def containers : List Container := [containerX, containerY, containerZ]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  (containers.map (fun c => containerProbability * greenProbability c)).sum = 8 / 15 := by
  sorry

end green_ball_probability_l1019_101944


namespace nancy_tortilla_chips_l1019_101977

/-- Nancy's tortilla chip distribution problem -/
theorem nancy_tortilla_chips : ∀ (initial brother sister : ℕ),
  initial = 22 →
  brother = 7 →
  sister = 5 →
  initial - (brother + sister) = 10 := by
  sorry

end nancy_tortilla_chips_l1019_101977


namespace inequality_proof_l1019_101946

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_sum : x + y + z = 3 * Real.sqrt 3) :
  (x^2 / (x + 2*y + 3*z)) + (y^2 / (y + 2*z + 3*x)) + (z^2 / (z + 2*x + 3*y)) ≥ Real.sqrt 3 / 2 := by
  sorry

#check inequality_proof

end inequality_proof_l1019_101946


namespace inequality_solution_set_l1019_101954

theorem inequality_solution_set (x : ℝ) :
  (x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2 := by
  sorry

end inequality_solution_set_l1019_101954


namespace quadratic_root_property_l1019_101927

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 50 = 0 → a^3 - 51*a = 50 := by
  sorry

end quadratic_root_property_l1019_101927


namespace andrew_eggs_l1019_101950

/-- The number of eggs Andrew ends up with after buying more -/
def total_eggs (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem: Andrew ends up with 70 eggs when starting with 8 and buying 62 more -/
theorem andrew_eggs : total_eggs 8 62 = 70 := by
  sorry

end andrew_eggs_l1019_101950


namespace football_players_count_l1019_101902

theorem football_players_count (total : ℕ) (basketball : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 22)
  (h2 : basketball = 13)
  (h3 : neither = 3)
  (h4 : both = 18) :
  total - neither - (basketball - both) = 19 :=
by sorry

end football_players_count_l1019_101902


namespace sugar_recipe_reduction_l1019_101997

theorem sugar_recipe_reduction : 
  (3 + 3 / 4 : ℚ) / 3 = 1 + 1 / 4 := by sorry

end sugar_recipe_reduction_l1019_101997


namespace long_letter_time_ratio_l1019_101930

/-- Represents the letter writing schedule and times for Steve --/
structure LetterWriting where
  days_between_letters : ℕ
  regular_letter_time : ℕ
  time_per_page : ℕ
  long_letter_time : ℕ
  total_pages_per_month : ℕ

/-- Calculates the ratio of time spent per page for the long letter compared to a regular letter --/
def time_ratio (lw : LetterWriting) : ℚ :=
  let regular_letters_per_month := 30 / lw.days_between_letters
  let pages_per_regular_letter := lw.regular_letter_time / lw.time_per_page
  let regular_letter_pages := regular_letters_per_month * pages_per_regular_letter
  let long_letter_pages := lw.total_pages_per_month - regular_letter_pages
  let long_letter_time_per_page := lw.long_letter_time / long_letter_pages
  long_letter_time_per_page / lw.time_per_page

/-- Theorem stating that the ratio of time spent per page for the long letter compared to a regular letter is 2:1 --/
theorem long_letter_time_ratio (lw : LetterWriting) 
  (h1 : lw.days_between_letters = 3)
  (h2 : lw.regular_letter_time = 20)
  (h3 : lw.time_per_page = 10)
  (h4 : lw.long_letter_time = 80)
  (h5 : lw.total_pages_per_month = 24) : 
  time_ratio lw = 2 := by
  sorry


end long_letter_time_ratio_l1019_101930


namespace power_of_two_inequality_l1019_101941

theorem power_of_two_inequality (k l m : ℕ) :
  2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 := by
  sorry

end power_of_two_inequality_l1019_101941


namespace smallest_z_value_l1019_101936

/-- Given an equation of consecutive perfect cubes, find the smallest possible value of the largest cube. -/
theorem smallest_z_value (u w x y z : ℕ) : 
  u^3 + w^3 + x^3 + y^3 = z^3 ∧ 
  (∃ k : ℕ, u = k ∧ w = k + 1 ∧ x = k + 2 ∧ y = k + 3 ∧ z = k + 4) ∧
  0 < u ∧ u < w ∧ w < x ∧ x < y ∧ y < z →
  z = 6 :=
by sorry

end smallest_z_value_l1019_101936


namespace square_sum_equals_two_l1019_101947

theorem square_sum_equals_two (x y : ℝ) 
  (h1 : x - y = -1) 
  (h2 : x * y = 1/2) : 
  x^2 + y^2 = 2 := by
sorry

end square_sum_equals_two_l1019_101947


namespace factorial_34_representation_l1019_101994

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def decimal_rep (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem factorial_34_representation (a b : ℕ) :
  decimal_rep (factorial 34) = [2, 9, 5, 2, 3, 2, 7, 9, 9, 0, 3, 9, a, 0, 4, 1, 4, 0, 8, 4, 7, 6, 1, 8, 6, 0, 9, 6, 4, 3, 5, b, 0, 0, 0, 0, 0, 0, 0] →
  a = 6 ∧ b = 2 := by
  sorry

end factorial_34_representation_l1019_101994


namespace complex_reciprocal_sum_magnitude_l1019_101962

theorem complex_reciprocal_sum_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 :=
by sorry

end complex_reciprocal_sum_magnitude_l1019_101962


namespace remainder_three_pow_244_mod_5_l1019_101932

theorem remainder_three_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end remainder_three_pow_244_mod_5_l1019_101932


namespace equation_solution_l1019_101916

theorem equation_solution : ∃ x : ℚ, 
  x = 81 / 16 ∧ 
  Real.sqrt x + 4 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 2*x :=
by
  sorry

end equation_solution_l1019_101916


namespace greatest_number_of_teams_l1019_101924

theorem greatest_number_of_teams (num_girls num_boys : ℕ) 
  (h_girls : num_girls = 40)
  (h_boys : num_boys = 32) :
  (∃ k : ℕ, k > 0 ∧ k ∣ num_girls ∧ k ∣ num_boys ∧ 
    ∀ m : ℕ, m > 0 → m ∣ num_girls → m ∣ num_boys → m ≤ k) ↔ 
  Nat.gcd num_girls num_boys = 8 :=
by sorry

end greatest_number_of_teams_l1019_101924


namespace quadratic_roots_imply_ratio_l1019_101964

theorem quadratic_roots_imply_ratio (a b : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x = -1/2 ∧ y = 1/3 ∧ a * x^2 + b * x + 2 = 0 ∧ a * y^2 + b * y + 2 = 0) →
  (a - b) / a = 5/6 := by
  sorry

end quadratic_roots_imply_ratio_l1019_101964


namespace bmw_sales_count_l1019_101926

-- Define the total number of cars sold
def total_cars : ℕ := 300

-- Define the percentages of non-BMW cars sold
def volkswagen_percent : ℚ := 10/100
def toyota_percent : ℚ := 25/100
def acura_percent : ℚ := 20/100

-- Define the theorem
theorem bmw_sales_count :
  let non_bmw_percent : ℚ := volkswagen_percent + toyota_percent + acura_percent
  let bmw_percent : ℚ := 1 - non_bmw_percent
  (bmw_percent * total_cars : ℚ) = 135 := by sorry

end bmw_sales_count_l1019_101926


namespace y_relationship_l1019_101986

/-- The function f(x) = -x² + 5 -/
def f (x : ℝ) : ℝ := -x^2 + 5

/-- y₁ is the y-coordinate of the point (-4, y₁) on the graph of f -/
def y₁ : ℝ := f (-4)

/-- y₂ is the y-coordinate of the point (-1, y₂) on the graph of f -/
def y₂ : ℝ := f (-1)

/-- y₃ is the y-coordinate of the point (2, y₃) on the graph of f -/
def y₃ : ℝ := f 2

theorem y_relationship : y₂ > y₃ ∧ y₃ > y₁ := by sorry

end y_relationship_l1019_101986


namespace quadratic_solution_property_l1019_101953

theorem quadratic_solution_property : ∀ d e : ℝ,
  (4 * d^2 + 8 * d - 48 = 0) →
  (4 * e^2 + 8 * e - 48 = 0) →
  d ≠ e →
  (d - e)^2 + 4 = 68 := by
  sorry

end quadratic_solution_property_l1019_101953


namespace palic_function_is_quadratic_l1019_101959

-- Define the Palić function
def PalicFunction (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  Continuous f ∧
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

-- Define the theorem
theorem palic_function_is_quadratic 
  (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 ≠ 1) 
  (f : ℝ → ℝ) 
  (hf : PalicFunction f a b c) : 
  ∃ A B C : ℝ, ∀ x : ℝ, f x = A * x^2 + B * x + C := by
sorry

end palic_function_is_quadratic_l1019_101959


namespace range_of_m_l1019_101999

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
  (∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0) →
  ¬(∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  (∃ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 = 0) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by sorry

end range_of_m_l1019_101999


namespace marble_bag_problem_l1019_101985

theorem marble_bag_problem (red blue : ℕ) (p : ℚ) (total : ℕ) : 
  red = 12 →
  blue = 8 →
  p = 81 / 256 →
  (((total - red : ℚ) / total) ^ 4 = p) →
  total = 48 :=
by sorry

end marble_bag_problem_l1019_101985


namespace least_subtrahend_proof_l1019_101955

/-- The product of the first four prime numbers -/
def product_of_first_four_primes : ℕ := 2 * 3 * 5 * 7

/-- The original number from which we subtract -/
def original_number : ℕ := 427751

/-- The least number to be subtracted -/
def least_subtrahend : ℕ := 91

theorem least_subtrahend_proof :
  (∀ k : ℕ, k < least_subtrahend → ¬((original_number - k) % product_of_first_four_primes = 0)) ∧
  ((original_number - least_subtrahend) % product_of_first_four_primes = 0) :=
sorry

end least_subtrahend_proof_l1019_101955


namespace opposite_reciprocal_expression_zero_l1019_101978

theorem opposite_reciprocal_expression_zero
  (a b c d : ℝ)
  (h1 : a = -b)
  (h2 : c = 1 / d)
  : 2 * c - a - 2 / d - b = 0 := by
  sorry

end opposite_reciprocal_expression_zero_l1019_101978


namespace zara_goats_l1019_101949

/-- The number of cows Zara bought -/
def num_cows : ℕ := 24

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of groups for transportation -/
def num_groups : ℕ := 3

/-- The number of animals per group -/
def animals_per_group : ℕ := 48

/-- The total number of animals -/
def total_animals : ℕ := num_groups * animals_per_group

/-- The number of goats Zara owns -/
def num_goats : ℕ := total_animals - (num_cows + num_sheep)

theorem zara_goats : num_goats = 113 := by
  sorry

end zara_goats_l1019_101949


namespace probability_independent_of_radius_constant_probability_l1019_101974

-- Define a circular dartboard
structure Dartboard where
  radius : ℝ
  radius_pos : radius > 0

-- Define the probability function
def probability_closer_to_center (d : Dartboard) : ℝ := 0.25

-- Theorem statement
theorem probability_independent_of_radius (d : Dartboard) :
  probability_closer_to_center d = 0.25 := by
  sorry

-- The distance from the thrower is not relevant to the probability,
-- but we include it to match the original problem description
def distance_from_thrower : ℝ := 20

-- Theorem stating that the probability is constant regardless of radius
theorem constant_probability (d1 d2 : Dartboard) :
  probability_closer_to_center d1 = probability_closer_to_center d2 := by
  sorry

end probability_independent_of_radius_constant_probability_l1019_101974


namespace side_length_equation_l1019_101965

/-- Rectangle ABCD with equilateral triangles AEF and XYZ -/
structure SpecialRectangle where
  /-- Length of rectangle ABCD -/
  length : ℝ
  /-- Width of rectangle ABCD -/
  width : ℝ
  /-- Point E on BC such that BE = EC -/
  E : ℝ × ℝ
  /-- Point F on CD -/
  F : ℝ × ℝ
  /-- Side length of equilateral triangle XYZ -/
  s : ℝ
  /-- Rectangle ABCD has length 2 and width 1 -/
  length_eq : length = 2
  /-- Rectangle ABCD has length 2 and width 1 -/
  width_eq : width = 1
  /-- BE = EC = 1 -/
  BE_eq_EC : E.1 = 1
  /-- Angle AEF is 60 degrees -/
  angle_AEF : Real.cos (60 * π / 180) = 1 / 2
  /-- Triangle AEF is equilateral -/
  AEF_equilateral : (E.1 - 0)^2 + (E.2 - 0)^2 = (F.1 - E.1)^2 + (F.2 - E.2)^2
  /-- XY is parallel to AB -/
  XY_parallel_AB : s ≤ width

theorem side_length_equation (r : SpecialRectangle) :
  r.s^2 + 4 * r.s - 8 / Real.sqrt 3 = 0 := by
  sorry

end side_length_equation_l1019_101965


namespace line_equation_through_points_l1019_101960

/-- Given two points (m, n) and (m + 3, n + 9) in the coordinate plane,
    prove that the equation y = 3x + (n - 3m) represents the line passing through these points. -/
theorem line_equation_through_points (m n : ℝ) :
  ∀ x y : ℝ, y = 3 * x + (n - 3 * m) ↔ (∃ t : ℝ, x = m + 3 * t ∧ y = n + 9 * t) :=
by sorry

end line_equation_through_points_l1019_101960


namespace sum_of_corners_is_164_l1019_101909

/-- Represents a square on the checkerboard -/
structure Square where
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- Function to get the number in a given square -/
def getNumber (s : Square) : Nat :=
  (s.row - 1) * boardSize + s.col

/-- The four corner squares of the board -/
def corners : List Square := [
  { row := 1, col := 1 },       -- Top left
  { row := 1, col := boardSize },  -- Top right
  { row := boardSize, col := 1 },  -- Bottom left
  { row := boardSize, col := boardSize }  -- Bottom right
]

/-- Theorem stating that the sum of numbers in the corners is 164 -/
theorem sum_of_corners_is_164 :
  (corners.map getNumber).sum = 164 := by sorry

end sum_of_corners_is_164_l1019_101909


namespace expression_simplification_l1019_101940

theorem expression_simplification (m : ℝ) (h : m ≠ 2) :
  (m + 2 - 5 / (m - 2)) / ((m - 3) / (2 * m - 4)) = 2 * m + 6 := by
  sorry

end expression_simplification_l1019_101940


namespace star_commutative_star_associative_star_identity_star_not_distributive_l1019_101939

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Associativity
theorem star_associative : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := by sorry

-- Identity element
theorem star_identity : ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x := by sorry

-- Not distributive over addition
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

end star_commutative_star_associative_star_identity_star_not_distributive_l1019_101939


namespace binary_to_decimal_101001_l1019_101908

/-- Converts a list of binary digits to its decimal representation -/
def binaryToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binaryNumber : List Nat := [1, 0, 1, 0, 0, 1]

/-- Theorem stating that the binary number 101001 is equal to the decimal number 41 -/
theorem binary_to_decimal_101001 :
  binaryToDecimal binaryNumber = 41 := by
  sorry

#eval binaryToDecimal binaryNumber

end binary_to_decimal_101001_l1019_101908


namespace solution_set_equivalence_l1019_101987

def solution_set : Set ℝ := {x | x ≤ -5/2}

def inequality (x : ℝ) : Prop := |x - 2| + |x + 3| ≥ 4

theorem solution_set_equivalence :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end solution_set_equivalence_l1019_101987


namespace ellipse_condition_l1019_101981

/-- The equation of the curve -/
def curve_equation (x y c : ℝ) : Prop :=
  9 * x^2 + y^2 + 54 * x - 8 * y = c

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (c : ℝ) : Prop :=
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, curve_equation x y c ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem: The curve is a non-degenerate ellipse if and only if c > -97 -/
theorem ellipse_condition (c : ℝ) :
  is_non_degenerate_ellipse c ↔ c > -97 := by
  sorry

end ellipse_condition_l1019_101981


namespace art_students_count_l1019_101970

theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 20)
  (h3 : both = 10)
  (h4 : neither = 470) :
  ∃ art : ℕ, art = 20 ∧ 
    total = (music - both) + (art - both) + both + neither :=
by sorry

end art_students_count_l1019_101970


namespace number_equation_solution_l1019_101901

theorem number_equation_solution : 
  ∃ x : ℝ, 0.4 * x + 60 = x ∧ x = 100 := by sorry

end number_equation_solution_l1019_101901


namespace median_is_5_probability_l1019_101917

def classCount : ℕ := 9
def selectedClassCount : ℕ := 5
def medianClassNumber : ℕ := 5

def probabilityMedianIs5 : ℚ :=
  (Nat.choose 4 2 * Nat.choose 4 2) / Nat.choose classCount selectedClassCount

theorem median_is_5_probability :
  probabilityMedianIs5 = 2 / 7 := by
  sorry

end median_is_5_probability_l1019_101917


namespace eight_balls_distribution_l1019_101920

/-- The number of ways to distribute n distinct balls into 3 boxes,
    where box i contains at least i balls -/
def distribution_count (n : ℕ) : ℕ := sorry

/-- Theorem stating that there are 2268 ways to distribute 8 distinct balls
    into 3 boxes numbered 1, 2, and 3, where each box i contains at least i balls -/
theorem eight_balls_distribution : distribution_count 8 = 2268 := by sorry

end eight_balls_distribution_l1019_101920


namespace inequality_not_always_hold_l1019_101900

theorem inequality_not_always_hold (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ c : ℝ, ¬(a * c > b * c) :=
by
  sorry

end inequality_not_always_hold_l1019_101900


namespace right_triangle_area_right_triangle_area_proof_l1019_101992

/-- The area of a right triangle with sides of length 8 and 3 is 12 -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun side1 side2 area =>
    side1 = 8 ∧ side2 = 3 ∧ area = (1 / 2) * side1 * side2 → area = 12

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 8 3 12 := by
  sorry

end right_triangle_area_right_triangle_area_proof_l1019_101992


namespace symmetric_circle_equation_l1019_101910

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetric circle C
def symmetric_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- Theorem stating that the symmetric circle C has the equation x^2 + (y + 1)^2 = 1
theorem symmetric_circle_equation :
  ∀ x y : ℝ,
  (∃ x' y' : ℝ, original_circle x' y' ∧ 
   symmetry_line ((x + x') / 2) ((y + y') / 2)) →
  symmetric_circle x y :=
sorry

end symmetric_circle_equation_l1019_101910


namespace bills_double_pay_threshold_l1019_101931

/-- Proves that Bill starts getting paid double after 40 hours -/
theorem bills_double_pay_threshold (base_rate : ℝ) (double_rate : ℝ) (total_hours : ℝ) (total_pay : ℝ)
  (h1 : base_rate = 20)
  (h2 : double_rate = 2 * base_rate)
  (h3 : total_hours = 50)
  (h4 : total_pay = 1200) :
  ∃ x : ℝ, x = 40 ∧ base_rate * x + double_rate * (total_hours - x) = total_pay :=
by
  sorry

end bills_double_pay_threshold_l1019_101931


namespace surveyed_not_population_l1019_101989

/-- Represents the total number of students in the seventh grade. -/
def total_students : ℕ := 800

/-- Represents the number of students surveyed. -/
def surveyed_students : ℕ := 200

/-- Represents whether a given number of students constitutes the entire population. -/
def is_population (n : ℕ) : Prop := n = total_students

/-- Theorem stating that the surveyed students do not constitute the entire population. -/
theorem surveyed_not_population : ¬(is_population surveyed_students) := by
  sorry

end surveyed_not_population_l1019_101989


namespace ant_movement_theorem_l1019_101998

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the movement of an ant --/
structure AntMovement where
  seconds : ℕ
  unitPerSecond : ℝ

/-- Calculates the expected area of the convex quadrilateral formed by ants --/
def expectedArea (rect : Rectangle) (movement : AntMovement) : ℝ :=
  (rect.length - 2 * movement.seconds * movement.unitPerSecond) *
  (rect.width - 2 * movement.seconds * movement.unitPerSecond)

/-- Theorem statement for the ant movement problem --/
theorem ant_movement_theorem (rect : Rectangle) (movement : AntMovement) :
  rect.length = 20 ∧ rect.width = 23 ∧ movement.seconds = 10 ∧ movement.unitPerSecond = 0.5 →
  expectedArea rect movement = 130 := by
  sorry


end ant_movement_theorem_l1019_101998


namespace ellipse_intersection_dot_product_l1019_101956

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci of the ellipse
def focus_1 : ℝ × ℝ := (1, 0)
def focus_2 : ℝ × ℝ := (-1, 0)

-- Define a line passing through a focus at 45°
def line_through_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - f.2 = (x - f.1)

-- Define the intersection points
def intersection_points (f : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | is_on_ellipse p.1 p.2 ∧ line_through_focus f p.1 p.2}

-- Theorem statement
theorem ellipse_intersection_dot_product :
  ∀ (f : ℝ × ℝ) (A B : ℝ × ℝ),
    (f = focus_1 ∨ f = focus_2) →
    A ∈ intersection_points f →
    B ∈ intersection_points f →
    A ≠ B →
    A.1 * B.1 + A.2 * B.2 = -1/3 :=
sorry

end ellipse_intersection_dot_product_l1019_101956


namespace sum_of_squared_coefficients_is_3148_l1019_101961

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 5 * (x^3 - 3*x^2 + 3) - 9 * (x^4 - 4*x^2 + 4)

/-- The sum of squares of coefficients of the fully simplified expression -/
def sum_of_squared_coefficients : ℝ := 3148

/-- Theorem stating that the sum of squares of coefficients of the fully simplified expression is 3148 -/
theorem sum_of_squared_coefficients_is_3148 :
  ∃ (a b c d : ℝ), ∀ (x : ℝ), 
    expression x = a*x^4 + b*x^3 + c*x^2 + d ∧
    a^2 + b^2 + c^2 + d^2 = sum_of_squared_coefficients :=
sorry

end sum_of_squared_coefficients_is_3148_l1019_101961


namespace isosceles_trapezoid_diagonal_l1019_101918

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  side : ℝ

/-- The diagonal of an isosceles trapezoid -/
def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specified isosceles trapezoid is 13 units -/
theorem isosceles_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { base1 := 24, base2 := 12, side := 13 }
  diagonal t = 13 := by
  sorry

end isosceles_trapezoid_diagonal_l1019_101918


namespace triangle_parallelogram_altitude_relation_l1019_101996

theorem triangle_parallelogram_altitude_relation 
  (base : ℝ) 
  (triangle_area parallelogram_area : ℝ) 
  (triangle_altitude parallelogram_altitude : ℝ) 
  (h1 : triangle_area = parallelogram_area) 
  (h2 : parallelogram_altitude = 100) 
  (h3 : triangle_area = 1/2 * base * triangle_altitude) 
  (h4 : parallelogram_area = base * parallelogram_altitude) : 
  triangle_altitude = 200 := by
sorry

end triangle_parallelogram_altitude_relation_l1019_101996


namespace remainder_2022_power_mod_11_l1019_101971

theorem remainder_2022_power_mod_11 : 2022^(2022^2022) ≡ 5 [ZMOD 11] := by
  sorry

end remainder_2022_power_mod_11_l1019_101971


namespace solve_equation_l1019_101995

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 5

-- State the theorem
theorem solve_equation (a : ℝ) (ha : a > 0) (h : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end solve_equation_l1019_101995


namespace tens_digit_of_8_power_23_l1019_101937

theorem tens_digit_of_8_power_23 : ∃ n : ℕ, 8^23 = 10 * n + 12 :=
sorry

end tens_digit_of_8_power_23_l1019_101937


namespace definite_integral_2x_plus_exp_l1019_101915

theorem definite_integral_2x_plus_exp : ∫ x in (0:ℝ)..1, (2 * x + Real.exp x) = Real.exp 1 - 1 := by
  sorry

end definite_integral_2x_plus_exp_l1019_101915


namespace arithmetic_sequence_with_geometric_subset_l1019_101951

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem arithmetic_sequence_with_geometric_subset (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 = 1 →
  is_geometric_sequence (a 1) (a 3) (a 9) →
  (∀ n : ℕ, a n = n) ∨ (∀ n : ℕ, a n = 1) :=
sorry

end arithmetic_sequence_with_geometric_subset_l1019_101951


namespace total_hours_worked_l1019_101933

/-- 
Given a person works 8 hours per day for 4 days, 
prove that the total number of hours worked is 32.
-/
theorem total_hours_worked (hours_per_day : ℕ) (days_worked : ℕ) : 
  hours_per_day = 8 → days_worked = 4 → hours_per_day * days_worked = 32 := by
sorry

end total_hours_worked_l1019_101933


namespace bus_stop_speed_fraction_l1019_101906

theorem bus_stop_speed_fraction (usual_time normal_delay : ℕ) (fraction : ℚ) : 
  usual_time = 28 →
  normal_delay = 7 →
  fraction * (usual_time + normal_delay) = usual_time →
  fraction = 4 / 5 := by
sorry

end bus_stop_speed_fraction_l1019_101906


namespace f_2023_of_2_eq_one_seventh_l1019_101935

-- Define the function f
def f (x : ℚ) : ℚ := (1 + x) / (1 - 3*x)

-- Define f_n recursively
def f_n : ℕ → (ℚ → ℚ)
  | 0 => f
  | 1 => λ x => f (f x)
  | (n+2) => λ x => f (f_n (n+1) x)

-- Theorem statement
theorem f_2023_of_2_eq_one_seventh : f_n 2023 2 = 1/7 := by sorry

end f_2023_of_2_eq_one_seventh_l1019_101935
