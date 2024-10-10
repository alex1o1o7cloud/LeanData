import Mathlib

namespace two_successes_in_four_trials_l599_59953

def probability_of_two_successes_in_four_trials (p : ℝ) : ℝ :=
  6 * p^2 * (1 - p)^2

theorem two_successes_in_four_trials :
  probability_of_two_successes_in_four_trials 0.6 = 0.3456 := by
  sorry

end two_successes_in_four_trials_l599_59953


namespace zero_point_in_interval_l599_59954

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 0 :=
by sorry

end zero_point_in_interval_l599_59954


namespace min_participants_quiz_l599_59915

-- Define the number of correct answers for each question
def correct_q1 : ℕ := 90
def correct_q2 : ℕ := 50
def correct_q3 : ℕ := 40
def correct_q4 : ℕ := 20

-- Define the maximum number of questions a participant can answer correctly
def max_correct_per_participant : ℕ := 2

-- Define the total number of correct answers
def total_correct_answers : ℕ := correct_q1 + correct_q2 + correct_q3 + correct_q4

-- Theorem stating the minimum number of participants
theorem min_participants_quiz : 
  ∀ n : ℕ, 
  (n * max_correct_per_participant ≥ total_correct_answers) → 
  (∀ m : ℕ, m < n → m * max_correct_per_participant < total_correct_answers) → 
  n = 100 :=
by sorry

end min_participants_quiz_l599_59915


namespace train_length_l599_59925

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time : ℝ) (h1 : speed_kmh = 90) (h2 : cross_time = 9) :
  speed_kmh * (1000 / 3600) * cross_time = 225 :=
by sorry

end train_length_l599_59925


namespace phantom_ink_problem_l599_59961

/-- The cost of a single black printer ink -/
def black_ink_cost : ℕ := 11

/-- The amount Phantom's mom gave him -/
def initial_amount : ℕ := 50

/-- The number of black printer inks bought -/
def black_ink_count : ℕ := 2

/-- The number of red printer inks bought -/
def red_ink_count : ℕ := 3

/-- The cost of each red printer ink -/
def red_ink_cost : ℕ := 15

/-- The number of yellow printer inks bought -/
def yellow_ink_count : ℕ := 2

/-- The cost of each yellow printer ink -/
def yellow_ink_cost : ℕ := 13

/-- The additional amount Phantom needs -/
def additional_amount : ℕ := 43

theorem phantom_ink_problem :
  black_ink_cost * black_ink_count +
  red_ink_cost * red_ink_count +
  yellow_ink_cost * yellow_ink_count =
  initial_amount + additional_amount :=
sorry

end phantom_ink_problem_l599_59961


namespace exam_total_boys_l599_59938

theorem exam_total_boys (average_all : ℚ) (average_passed : ℚ) (average_failed : ℚ) 
  (passed_count : ℕ) : 
  average_all = 40 ∧ average_passed = 39 ∧ average_failed = 15 ∧ passed_count = 125 → 
  ∃ (total_count : ℕ), total_count = 120 ∧ 
    average_all * total_count = average_passed * passed_count + 
      average_failed * (total_count - passed_count) :=
by sorry

end exam_total_boys_l599_59938


namespace logarithmic_equation_solution_l599_59918

theorem logarithmic_equation_solution (x : ℝ) : 
  x > 0 → 
  (7.3113 * (Real.log 4 / Real.log x) + 
   2 * (Real.log 4 / Real.log (4 * x)) + 
   3 * (Real.log 4 / Real.log (16 * x)) = 0) ↔ 
  (x = 1/2 ∨ x = 1/8) := by
  sorry

end logarithmic_equation_solution_l599_59918


namespace barn_painting_area_l599_59923

theorem barn_painting_area (width length height : ℝ) 
  (h_width : width = 10)
  (h_length : length = 13)
  (h_height : height = 5) :
  2 * (width * height + length * height) + width * length = 590 :=
by sorry

end barn_painting_area_l599_59923


namespace betty_beads_l599_59941

theorem betty_beads (red blue green : ℕ) : 
  (5 * blue = 3 * red) →
  (5 * green = 2 * red) →
  (red = 50) →
  (blue + green = 50) := by
sorry

end betty_beads_l599_59941


namespace quadratic_function_k_range_l599_59998

/-- Given a quadratic function f(x) = 4x^2 - kx - 8 with no maximum or minimum at (5, 20),
    prove that the range of k is k ≤ 40 or k ≥ 160. -/
theorem quadratic_function_k_range (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - k * x - 8
  (∀ x, f x ≠ f 5 ∨ (∃ y, y ≠ 5 ∧ f y = f 5)) →
  f 5 = 20 →
  k ≤ 40 ∨ k ≥ 160 := by
sorry

end quadratic_function_k_range_l599_59998


namespace product_equals_fraction_l599_59964

/-- The decimal representation of a real number with digits 1, 4, 5 repeating after the decimal point -/
def repeating_decimal : ℚ := 145 / 999

/-- The product of the repeating decimal and 11 -/
def product : ℚ := 11 * repeating_decimal

theorem product_equals_fraction : product = 1595 / 999 := by
  sorry

end product_equals_fraction_l599_59964


namespace unique_positive_integer_solution_l599_59934

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 + 2 * x = 3528 := by
  sorry

end unique_positive_integer_solution_l599_59934


namespace janet_stuffies_l599_59986

theorem janet_stuffies (x : ℚ) : 
  let total := x
  let kept := (3 / 7) * total
  let distributed := total - kept
  let ratio_sum := 3 + 4 + 2 + 1 + 5
  let janet_part := 1
  (janet_part / ratio_sum) * distributed = (4 * x) / 105 := by
sorry

end janet_stuffies_l599_59986


namespace remainder_theorem_l599_59947

/-- The polynomial P(x) = 5x^4 - 13x^3 + 3x^2 - x + 15 -/
def P (x : ℝ) : ℝ := 5*x^4 - 13*x^3 + 3*x^2 - x + 15

/-- The divisor polynomial d(x) = 3x - 9 -/
def d (x : ℝ) : ℝ := 3*x - 9

/-- Theorem stating that the remainder when P(x) is divided by d(x) is 93 -/
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, P x = d x * q x + 93 := by
  sorry

end remainder_theorem_l599_59947


namespace intersection_when_m_3_union_equals_A_iff_l599_59966

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem for part 1
theorem intersection_when_m_3 : 
  A ∩ B 3 = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for part 2
theorem union_equals_A_iff (m : ℝ) : 
  A ∪ B m = A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

end intersection_when_m_3_union_equals_A_iff_l599_59966


namespace largest_n_satisfying_inequality_l599_59912

theorem largest_n_satisfying_inequality : 
  ∀ n : ℤ, (1/4 : ℚ) + (n : ℚ)/6 < 3/2 ↔ n ≤ 7 :=
by sorry

end largest_n_satisfying_inequality_l599_59912


namespace calculate_expression_l599_59946

theorem calculate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^3) = (3 / 4) * y^2 := by
  sorry

end calculate_expression_l599_59946


namespace sqrt_three_times_sqrt_six_equals_three_sqrt_two_l599_59950

theorem sqrt_three_times_sqrt_six_equals_three_sqrt_two :
  Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_three_times_sqrt_six_equals_three_sqrt_two_l599_59950


namespace ratio_difference_l599_59973

theorem ratio_difference (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := by
  sorry

end ratio_difference_l599_59973


namespace no_finite_vector_set_with_equal_sums_property_l599_59905

theorem no_finite_vector_set_with_equal_sums_property (n : ℕ) :
  ¬ ∃ (S : Finset (ℝ × ℝ)),
    (S.card = n) ∧
    (∀ (a b : ℝ × ℝ), a ∈ S → b ∈ S → a ≠ b →
      ∃ (c d : ℝ × ℝ), c ∈ S ∧ d ∈ S ∧ c ≠ d ∧ c ≠ a ∧ c ≠ b ∧ d ≠ a ∧ d ≠ b ∧
        a.1 + b.1 = c.1 + d.1 ∧ a.2 + b.2 = c.2 + d.2) :=
by sorry

end no_finite_vector_set_with_equal_sums_property_l599_59905


namespace terminating_decimal_of_7_over_200_l599_59929

theorem terminating_decimal_of_7_over_200 : 
  ∃ (n : ℕ) (d : ℕ+), (7 : ℚ) / 200 = (n : ℚ) / d ∧ (n : ℚ) / d = 0.028 := by
  sorry

end terminating_decimal_of_7_over_200_l599_59929


namespace pizza_slices_per_person_l599_59944

theorem pizza_slices_per_person 
  (small_pizza_slices : ℕ) 
  (large_pizza_slices : ℕ) 
  (slices_eaten_per_person : ℕ) 
  (num_people : ℕ) :
  small_pizza_slices = 8 →
  large_pizza_slices = 14 →
  slices_eaten_per_person = 9 →
  num_people = 2 →
  (small_pizza_slices + large_pizza_slices - slices_eaten_per_person * num_people) / num_people = 2 := by
sorry

end pizza_slices_per_person_l599_59944


namespace a_power_b_is_one_fourth_l599_59971

theorem a_power_b_is_one_fourth (a b : ℝ) (h : (a + b)^2 + |b + 2| = 0) : a^b = 1/4 := by
  sorry

end a_power_b_is_one_fourth_l599_59971


namespace unit_digit_15_power_100_l599_59922

theorem unit_digit_15_power_100 : (15 ^ 100) % 10 = 5 := by
  sorry

end unit_digit_15_power_100_l599_59922


namespace more_karabases_than_barabases_l599_59965

/-- Represents the inhabitants of Perra-Terra -/
inductive Inhabitant
| Karabas
| Barabas

/-- The number of acquaintances each type of inhabitant has -/
def acquaintances (i : Inhabitant) : Nat × Nat :=
  match i with
  | Inhabitant.Karabas => (6, 9)  -- (Other Karabases, Barabases)
  | Inhabitant.Barabas => (10, 7) -- (Karabases, Other Barabases)

theorem more_karabases_than_barabases (K B : Nat) 
  (hK : K > 0) (hB : B > 0) 
  (h_acquaintances : K * (acquaintances Inhabitant.Karabas).2 = B * (acquaintances Inhabitant.Barabas).1) :
  K > B := by
  sorry

#check more_karabases_than_barabases

end more_karabases_than_barabases_l599_59965


namespace otimes_inequality_range_l599_59995

/-- Custom operation ⊗ on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' for which the inequality holds for all real x -/
theorem otimes_inequality_range (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) ↔ a ∈ Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) :=
sorry

end otimes_inequality_range_l599_59995


namespace functional_equation_l599_59997

theorem functional_equation (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x - y) = f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
  sorry

end functional_equation_l599_59997


namespace quadratic_inequality_equivalence_l599_59996

theorem quadratic_inequality_equivalence (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k-2)*x - k + 8 ≥ 0) ↔ 
  (k ≥ -2 * Real.sqrt 7 ∧ k ≤ 2 * Real.sqrt 7) :=
sorry

end quadratic_inequality_equivalence_l599_59996


namespace batsman_average_theorem_l599_59993

def batting_average (innings : ℕ) (total_runs : ℕ) : ℚ :=
  total_runs / innings

def revised_average (innings : ℕ) (total_runs : ℕ) (not_out : ℕ) : ℚ :=
  total_runs / (innings - not_out)

theorem batsman_average_theorem (total_runs_11 : ℕ) (innings : ℕ) (last_score : ℕ) (not_out : ℕ) :
  innings = 12 →
  last_score = 92 →
  not_out = 3 →
  batting_average innings (total_runs_11 + last_score) - batting_average (innings - 1) total_runs_11 = 2 →
  batting_average innings (total_runs_11 + last_score) = 70 ∧
  revised_average innings (total_runs_11 + last_score) not_out = 93.33 := by
  sorry

end batsman_average_theorem_l599_59993


namespace prime_divides_sum_of_squares_l599_59933

theorem prime_divides_sum_of_squares (p a b : ℤ) : 
  Prime p → p % 4 = 3 → (a^2 + b^2) % p = 0 → p ∣ a ∧ p ∣ b := by
  sorry

end prime_divides_sum_of_squares_l599_59933


namespace prob_select_boy_is_half_prob_same_gender_is_third_l599_59939

/-- The number of students in the class -/
def total_students : ℕ := 4

/-- The number of boys in the class -/
def num_boys : ℕ := 2

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The probability of selecting a boy when one student is randomly selected -/
def prob_select_boy : ℚ := num_boys / total_students

/-- The probability of selecting two students of the same gender when two students are randomly selected -/
def prob_same_gender : ℚ := 1 / 3

theorem prob_select_boy_is_half : prob_select_boy = 1 / 2 :=
sorry

theorem prob_same_gender_is_third : prob_same_gender = 1 / 3 :=
sorry

end prob_select_boy_is_half_prob_same_gender_is_third_l599_59939


namespace friend_distribution_l599_59936

/-- The number of ways to distribute n distinguishable items among k categories -/
def distribute (n k : ℕ) : ℕ := k ^ n

/-- The number of friends to be distributed -/
def num_friends : ℕ := 8

/-- The number of clubs available -/
def num_clubs : ℕ := 4

theorem friend_distribution :
  distribute num_friends num_clubs = 65536 := by
  sorry

end friend_distribution_l599_59936


namespace second_markdown_percentage_l599_59931

theorem second_markdown_percentage 
  (original_price : ℝ) 
  (first_markdown_percentage : ℝ) 
  (second_markdown_percentage : ℝ) 
  (h1 : first_markdown_percentage = 10)
  (h2 : (1 - first_markdown_percentage / 100) * (1 - second_markdown_percentage / 100) * original_price = 0.81 * original_price) :
  second_markdown_percentage = 10 := by
sorry

end second_markdown_percentage_l599_59931


namespace g_injective_on_restricted_domain_c_is_smallest_l599_59930

/-- The function g(x) = (x+3)^2 - 6 -/
def g (x : ℝ) : ℝ := (x + 3)^2 - 6

/-- c is the lower bound of the restricted domain -/
def c : ℝ := -3

theorem g_injective_on_restricted_domain :
  ∀ x y, x ≥ c → y ≥ c → g x = g y → x = y :=
sorry

theorem c_is_smallest :
  ∀ c' < c, ∃ x y, x ≥ c' ∧ y ≥ c' ∧ x ≠ y ∧ g x = g y :=
sorry

end g_injective_on_restricted_domain_c_is_smallest_l599_59930


namespace envelope_area_l599_59988

/-- The area of a rectangular envelope with width and height both 6 inches is 36 square inches. -/
theorem envelope_area (width height : ℝ) (h1 : width = 6) (h2 : height = 6) :
  width * height = 36 := by
  sorry

end envelope_area_l599_59988


namespace lemon_problem_l599_59942

theorem lemon_problem (levi jayden eli ian : ℕ) : 
  levi = 5 →
  jayden > levi →
  jayden * 3 = eli →
  eli * 2 = ian →
  levi + jayden + eli + ian = 115 →
  jayden - levi = 6 :=
by
  sorry

end lemon_problem_l599_59942


namespace unique_intersection_and_geometric_progression_l599_59987

noncomputable section

def f (x : ℝ) : ℝ := x / Real.exp x
def g (x : ℝ) : ℝ := Real.log x / x

theorem unique_intersection_and_geometric_progression :
  (∃! x : ℝ, f x = g x) ∧
  (∀ a : ℝ, 0 < a → a < Real.exp (-1) →
    (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
      f x₁ = a ∧ g x₂ = a ∧ f x₃ = a →
      ∃ r : ℝ, x₂ = x₁ * r ∧ x₃ = x₂ * r)) :=
sorry

end unique_intersection_and_geometric_progression_l599_59987


namespace envelope_difference_l599_59960

theorem envelope_difference (blue_envelopes : ℕ) (total_envelopes : ℕ) (yellow_envelopes : ℕ) :
  blue_envelopes = 10 →
  total_envelopes = 16 →
  yellow_envelopes < blue_envelopes →
  yellow_envelopes + blue_envelopes = total_envelopes →
  blue_envelopes - yellow_envelopes = 4 :=
by
  sorry

end envelope_difference_l599_59960


namespace sum_of_right_angles_l599_59957

/-- A rectangle has 4 right angles -/
def rectangle_right_angles : ℕ := 4

/-- A square has 4 right angles -/
def square_right_angles : ℕ := 4

/-- The sum of right angles in a rectangle and a square -/
def total_right_angles : ℕ := rectangle_right_angles + square_right_angles

theorem sum_of_right_angles : total_right_angles = 8 := by
  sorry

end sum_of_right_angles_l599_59957


namespace constant_speed_travel_time_l599_59958

/-- Given a constant speed, if a 120-mile trip takes 3 hours, then a 200-mile trip takes 5 hours. -/
theorem constant_speed_travel_time 
  (speed : ℝ) 
  (h₁ : speed > 0) 
  (h₂ : 120 / speed = 3) : 
  200 / speed = 5 := by
sorry

end constant_speed_travel_time_l599_59958


namespace hyperbola_asymptote_angle_l599_59990

/-- Given a hyperbola x^2 - y^2/b^2 = 1 with b > 1, the angle θ between its asymptotes is not 2arctan(b) -/
theorem hyperbola_asymptote_angle (b : ℝ) (h : b > 1) :
  let θ := Real.pi - 2 * Real.arctan b
  θ ≠ 2 * Real.arctan b :=
by sorry

end hyperbola_asymptote_angle_l599_59990


namespace sara_pumpkins_l599_59907

/-- The number of pumpkins Sara has now -/
def pumpkins_left : ℕ := 20

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The initial number of pumpkins Sara grew -/
def initial_pumpkins : ℕ := pumpkins_left + pumpkins_eaten

theorem sara_pumpkins : initial_pumpkins = 43 := by
  sorry

end sara_pumpkins_l599_59907


namespace prob_product_one_four_dice_l599_59983

/-- The number of sides on a standard die -/
def dieSides : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def probSingleDie : ℚ := 1 / dieSides

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability of rolling all ones on multiple dice -/
def probAllOnes : ℚ := probSingleDie ^ numDice

theorem prob_product_one_four_dice :
  probAllOnes = 1 / 1296 := by sorry

end prob_product_one_four_dice_l599_59983


namespace at_least_two_acute_angles_l599_59943

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define an acute angle
def is_acute (angle : ℝ) : Prop := angle < 90

-- Define the theorem
theorem at_least_two_acute_angles (t : Triangle) : 
  ∃ i j, i ≠ j ∧ is_acute (t.angles i) ∧ is_acute (t.angles j) :=
sorry

end at_least_two_acute_angles_l599_59943


namespace ticket_order_solution_l599_59924

/-- Represents the ticket order information -/
structure TicketOrder where
  childPrice : ℚ
  adultPrice : ℚ
  discountThreshold : ℕ
  discountRate : ℚ
  childrenExcess : ℕ
  totalBill : ℚ

/-- Calculates the number of adult and children tickets -/
def calculateTickets (order : TicketOrder) : ℕ × ℕ :=
  sorry

/-- Checks if the discount was applied -/
def wasDiscountApplied (order : TicketOrder) (adultTickets childTickets : ℕ) : Bool :=
  sorry

theorem ticket_order_solution (order : TicketOrder)
    (h1 : order.childPrice = 7.5)
    (h2 : order.adultPrice = 12)
    (h3 : order.discountThreshold = 20)
    (h4 : order.discountRate = 0.1)
    (h5 : order.childrenExcess = 8)
    (h6 : order.totalBill = 138) :
    let (adultTickets, childTickets) := calculateTickets order
    adultTickets = 4 ∧ childTickets = 12 ∧ ¬wasDiscountApplied order adultTickets childTickets :=
  sorry

end ticket_order_solution_l599_59924


namespace jacket_cost_is_30_l599_59984

/-- Represents the cost of clothing items in a discount store. -/
structure ClothingCost where
  sweater : ℝ
  jacket : ℝ

/-- Represents a shipment of clothing items. -/
structure Shipment where
  sweaters : ℕ
  jackets : ℕ
  totalCost : ℝ

/-- The conditions of the problem. -/
def problemConditions (cost : ClothingCost) : Prop :=
  ∃ (shipment1 shipment2 : Shipment),
    shipment1.sweaters = 10 ∧
    shipment1.jackets = 20 ∧
    shipment1.totalCost = 800 ∧
    shipment2.sweaters = 5 ∧
    shipment2.jackets = 15 ∧
    shipment2.totalCost = 550 ∧
    shipment1.totalCost = cost.sweater * shipment1.sweaters + cost.jacket * shipment1.jackets ∧
    shipment2.totalCost = cost.sweater * shipment2.sweaters + cost.jacket * shipment2.jackets

/-- The main theorem stating that under the given conditions, the cost of a jacket is $30. -/
theorem jacket_cost_is_30 :
  ∀ (cost : ClothingCost), problemConditions cost → cost.jacket = 30 := by
  sorry


end jacket_cost_is_30_l599_59984


namespace three_digit_powers_of_two_l599_59969

theorem three_digit_powers_of_two (n : ℕ) : 
  (∃ m : ℕ, 100 ≤ 2^m ∧ 2^m ≤ 999) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end three_digit_powers_of_two_l599_59969


namespace part_one_part_two_l599_59981

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Define propositions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ A a
def q (x : ℝ) : Prop := x ∈ B

-- Theorem for part I
theorem part_one (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := by
  sorry

-- Theorem for part II
theorem part_two (a : ℝ) (h : ∀ x, p x a → q x) : a ≤ 0 ∨ a ≥ 4 := by
  sorry

end part_one_part_two_l599_59981


namespace proposition_relationship_l599_59928

theorem proposition_relationship (a b : ℝ) :
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ ¬(a > b ∧ a⁻¹ > b⁻¹)) :=
by sorry

end proposition_relationship_l599_59928


namespace trailingZeroes_15_factorial_base12_l599_59980

/-- The number of trailing zeroes in the base 12 representation of 15! -/
def trailingZeroesBase12Factorial15 : ℕ :=
  min (Nat.factorial 15 / 12^5) 1

theorem trailingZeroes_15_factorial_base12 :
  trailingZeroesBase12Factorial15 = 5 := by
  sorry

end trailingZeroes_15_factorial_base12_l599_59980


namespace complex_number_equality_l599_59904

theorem complex_number_equality (z : ℂ) (h : z * Complex.I = 2 - 2 * Complex.I) : z = -2 - 2 * Complex.I := by
  sorry

end complex_number_equality_l599_59904


namespace two_smallest_solutions_l599_59916

def is_solution (k : ℕ) : Prop :=
  (Real.cos ((k^2 + 7^2 : ℝ) * Real.pi / 180))^2 = 1

def smallest_solutions : Prop :=
  (is_solution 31 ∧ is_solution 37) ∧
  ∀ k : ℕ, 0 < k ∧ k < 31 → ¬is_solution k

theorem two_smallest_solutions : smallest_solutions := by
  sorry

end two_smallest_solutions_l599_59916


namespace sqrt_inequality_not_arithmetic_sequence_l599_59992

-- Statement 1
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by sorry

-- Statement 2
theorem not_arithmetic_sequence : 
  ¬ ∃ (d k : ℝ), (k = 1 ∧ k + d = Real.sqrt 2 ∧ k + 2*d = 3) := by sorry

end sqrt_inequality_not_arithmetic_sequence_l599_59992


namespace equation_solution_l599_59991

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, Real.sqrt (39 - 6 * Real.sqrt 12) + Real.sqrt (k * x * (k * x + Real.sqrt 12) + 3) = 2 * k) ↔ 
  (k = 3 ∨ k = 6) :=
sorry

end equation_solution_l599_59991


namespace triangle_existence_and_area_l599_59900

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 6 ∧
  Real.sin t.B ^ 2 + Real.sin t.C ^ 2 = Real.sin t.A ^ 2 + (2 * Real.sqrt 3 / 3) * Real.sin t.A * Real.sin t.B * Real.sin t.C

-- Define the theorem
theorem triangle_existence_and_area (t : Triangle) :
  triangle_conditions t → t.b + t.c = 2 * Real.sqrt 3 →
  ∃ (area : ℝ), area = Real.sqrt 3 / 2 := by
  sorry

end triangle_existence_and_area_l599_59900


namespace sandy_spending_percentage_l599_59972

def initial_amount : ℝ := 300
def remaining_amount : ℝ := 210

theorem sandy_spending_percentage :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end sandy_spending_percentage_l599_59972


namespace last_five_digits_of_product_l599_59921

theorem last_five_digits_of_product : 
  (99 * 10101 * 111 * 1001001) % (100000 : ℕ) = 88889 := by
  sorry

end last_five_digits_of_product_l599_59921


namespace quadratic_vertex_l599_59903

/-- The quadratic function f(x) = 2(x - 4)^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * (x - 4)^2 + 5

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (4, 5)

theorem quadratic_vertex :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
by sorry

end quadratic_vertex_l599_59903


namespace sin_sum_to_product_l599_59949

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (9 * x) = 2 * Real.sin (6 * x) * Real.cos (3 * x) := by
  sorry

end sin_sum_to_product_l599_59949


namespace cubic_root_equation_solution_l599_59963

theorem cubic_root_equation_solution :
  ∃ x : ℝ, x = 1674 / 15 ∧ (30 * x + (30 * x + 27) ^ (1/3)) ^ (1/3) = 15 := by
sorry

end cubic_root_equation_solution_l599_59963


namespace range_of_a_l599_59979

-- Define the set of real numbers a that satisfy the condition
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, a * x^2 + a * x + 1 > 0}

-- Theorem stating that A is equal to the interval [0, 4)
theorem range_of_a : A = Set.Icc 0 (4 : ℝ) := by sorry

end range_of_a_l599_59979


namespace geometric_sequence_inequality_l599_59999

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a n > 0
  h2 : q ≠ 1
  h3 : ∀ n, a (n + 1) = q * a n

/-- The sum of the first and eighth terms is greater than the sum of the fourth and fifth terms -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.a 1 + seq.a 8 > seq.a 4 + seq.a 5 := by
  sorry

end geometric_sequence_inequality_l599_59999


namespace total_distinct_plants_l599_59955

-- Define the flower beds as finite sets
variable (A B C D : Finset ℕ)

-- Define the cardinalities of the sets
variable (hA : A.card = 550)
variable (hB : B.card = 500)
variable (hC : C.card = 400)
variable (hD : D.card = 350)

-- Define the intersections
variable (hAB : (A ∩ B).card = 60)
variable (hAC : (A ∩ C).card = 110)
variable (hAD : (A ∩ D).card = 70)
variable (hABC : (A ∩ B ∩ C).card = 30)

-- Define the empty intersections
variable (hBC : (B ∩ C).card = 0)
variable (hBD : (B ∩ D).card = 0)

-- State the theorem
theorem total_distinct_plants :
  (A ∪ B ∪ C ∪ D).card = 1590 :=
sorry

end total_distinct_plants_l599_59955


namespace sum_base8_327_73_l599_59908

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent. -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits. -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimalToBase8 (n / 8)

/-- The sum of 327₈ and 73₈ in base 8 is equal to 422₈. -/
theorem sum_base8_327_73 :
  decimalToBase8 (base8ToDecimal [3, 2, 7] + base8ToDecimal [7, 3]) = [4, 2, 2] := by
  sorry

end sum_base8_327_73_l599_59908


namespace courtyard_length_l599_59952

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) : 
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 24000 →
  (width * (num_bricks * brick_length * brick_width / width)) = 30 := by
sorry

end courtyard_length_l599_59952


namespace fluorescent_tubes_count_l599_59932

theorem fluorescent_tubes_count :
  ∀ (x y : ℕ),
  x + y = 13 →
  x / 3 + y / 2 = 5 →
  x = 9 :=
by
  sorry

end fluorescent_tubes_count_l599_59932


namespace polynomial_factorization_sum_l599_59967

theorem polynomial_factorization_sum (a b c : ℝ) : 
  (∀ x, x^2 + 17*x + 52 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 27 := by
sorry

end polynomial_factorization_sum_l599_59967


namespace coefficient_x_seven_l599_59909

theorem coefficient_x_seven (x : ℝ) :
  ∃ (a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ),
    (x + 1)^5 * (2*x - 1)^3 = a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧
    a₇ = 28 :=
by sorry

end coefficient_x_seven_l599_59909


namespace intersection_distance_squared_is_96_l599_59945

/-- The square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  let (x1, y1) := c1_center
  let (x2, y2) := c2_center
  -- Definition of the function, to be implemented
  0

/-- The theorem stating the square of the distance between intersection points -/
theorem intersection_distance_squared_is_96 :
  intersection_distance_squared (3, 2) (3, -4) 5 7 = 96 := by
  sorry

end intersection_distance_squared_is_96_l599_59945


namespace max_value_on_curve_l599_59902

theorem max_value_on_curve :
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4 →
  ∃ M : ℝ, M = 17 ∧ ∀ x' y' : ℝ, (x' - 1)^2 + (y' - 1)^2 = 4 → 3*x' + 4*y' ≤ M :=
by sorry

end max_value_on_curve_l599_59902


namespace johny_east_south_difference_l599_59911

/-- Represents Johny's travel distances in different directions -/
structure TravelDistances where
  south : ℝ
  east : ℝ
  north : ℝ

/-- Johny's travel conditions -/
def johny_travel : TravelDistances → Prop :=
  λ d => d.south = 40 ∧
         d.east > d.south ∧
         d.north = 2 * d.east ∧
         d.south + d.east + d.north = 220

/-- The theorem to prove -/
theorem johny_east_south_difference (d : TravelDistances) 
  (h : johny_travel d) : d.east - d.south = 40 :=
by
  sorry


end johny_east_south_difference_l599_59911


namespace mutually_exclusive_not_contradictory_l599_59906

structure Ball :=
  (color : String)

def Bag : Finset Ball := sorry

axiom bag_composition : 
  (Bag.filter (λ b => b.color = "red")).card = 2 ∧ 
  (Bag.filter (λ b => b.color = "black")).card = 2

def Draw : Finset Ball := sorry

axiom draw_size : Draw.card = 2

def exactly_one_black : Prop :=
  (Draw.filter (λ b => b.color = "black")).card = 1

def exactly_two_black : Prop :=
  (Draw.filter (λ b => b.color = "black")).card = 2

theorem mutually_exclusive_not_contradictory :
  (¬(exactly_one_black ∧ exactly_two_black)) ∧
  (∃ draw : Finset Ball, draw.card = 2 ∧ ¬exactly_one_black ∧ ¬exactly_two_black) :=
sorry

end mutually_exclusive_not_contradictory_l599_59906


namespace circle_circumference_l599_59978

/-- Given two circles with equal areas, where half the radius of one circle is 4.5,
    prove that the circumference of the other circle is 18π. -/
theorem circle_circumference (x y : ℝ) (harea : π * x^2 = π * y^2) (hy : y / 2 = 4.5) :
  2 * π * x = 18 * π := by
  sorry

end circle_circumference_l599_59978


namespace cos_120_degrees_l599_59989

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end cos_120_degrees_l599_59989


namespace condition_necessary_not_sufficient_l599_59901

/-- Definition of a geometric progression for three real numbers -/
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The condition b^2 = ac -/
def condition (a b c : ℝ) : Prop := b^2 = a * c

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ a b c : ℝ, is_geometric_progression a b c → condition a b c) ∧
  ¬(∀ a b c : ℝ, condition a b c → is_geometric_progression a b c) :=
sorry

end condition_necessary_not_sufficient_l599_59901


namespace intersection_point_l599_59968

theorem intersection_point (x y : ℝ) : 
  y = 4 * x - 32 ∧ y = -6 * x + 8 → (x, y) = (4, -16) :=
sorry

end intersection_point_l599_59968


namespace second_year_compound_interest_l599_59919

/-- Represents the compound interest for a given year -/
def CompoundInterest (principal : ℝ) (rate : ℝ) (year : ℕ) : ℝ :=
  principal * (1 + rate) ^ year - principal

/-- Theorem stating that given a 5% interest rate and a third-year compound interest of $1260,
    the second-year compound interest is $1200 -/
theorem second_year_compound_interest
  (principal : ℝ)
  (h1 : CompoundInterest principal 0.05 3 = 1260)
  (h2 : principal > 0) :
  CompoundInterest principal 0.05 2 = 1200 := by
  sorry


end second_year_compound_interest_l599_59919


namespace cookie_difference_l599_59935

/-- Proves that the difference between the number of cookies in 8 boxes and 9 bags is 33,
    given that each box contains 12 cookies and each bag contains 7 cookies. -/
theorem cookie_difference :
  let cookies_per_box : ℕ := 12
  let cookies_per_bag : ℕ := 7
  let num_boxes : ℕ := 8
  let num_bags : ℕ := 9
  (num_boxes * cookies_per_box) - (num_bags * cookies_per_bag) = 33 := by
  sorry

end cookie_difference_l599_59935


namespace complex_fraction_sum_l599_59976

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = -2 := by
  sorry

end complex_fraction_sum_l599_59976


namespace smallest_tangent_circle_l599_59970

/-- The line x + y - 2 = 0 -/
def line (x y : ℝ) : Prop := x + y - 2 = 0

/-- The curve x^2 + y^2 - 12x - 12y + 54 = 0 -/
def curve (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 12*y + 54 = 0

/-- The circle with center (6, 6) and radius 3√2 -/
def small_circle (x y : ℝ) : Prop := (x - 6)^2 + (y - 6)^2 = (3 * Real.sqrt 2)^2

/-- A circle is tangent to the line and curve if it touches them at exactly one point each -/
def is_tangent (circle line curve : ℝ → ℝ → Prop) : Prop := sorry

/-- A circle has the smallest radius if no other circle with a smaller radius is tangent to both the line and curve -/
def has_smallest_radius (circle line curve : ℝ → ℝ → Prop) : Prop := sorry

theorem smallest_tangent_circle :
  is_tangent small_circle line curve ∧ has_smallest_radius small_circle line curve := by sorry

end smallest_tangent_circle_l599_59970


namespace sum_of_repeating_decimals_l599_59994

/-- The sum of the repeating decimals 0.3̄ and 0.6̄ is equal to 1. -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), (∀ n : ℕ, x * 10^n - ⌊x * 10^n⌋ = 0.3) ∧ 
                (∀ n : ℕ, y * 10^n - ⌊y * 10^n⌋ = 0.6) ∧ 
                x + y = 1) := by
  sorry

end sum_of_repeating_decimals_l599_59994


namespace average_other_color_marbles_l599_59975

/-- Given a collection of marbles where 40% are clear, 20% are black, and the remainder are other colors,
    prove that when taking 5 marbles, the average number of marbles of other colors is 2. -/
theorem average_other_color_marbles
  (total : ℕ) -- Total number of marbles
  (clear : ℕ) -- Number of clear marbles
  (black : ℕ) -- Number of black marbles
  (other : ℕ) -- Number of other color marbles
  (h1 : clear = (40 * total) / 100) -- 40% are clear
  (h2 : black = (20 * total) / 100) -- 20% are black
  (h3 : other = total - clear - black) -- Remainder are other colors
  : (40 : ℚ) / 100 * 5 = 2 := by
  sorry

end average_other_color_marbles_l599_59975


namespace blueberries_for_pint_of_jam_l599_59917

/-- The number of blueberries needed to make a pint of blueberry jam -/
def blueberries_per_pint (blueberries_for_pies : ℕ) (num_pies : ℕ) : ℕ :=
  blueberries_for_pies / (num_pies * 2)

/-- Theorem stating the number of blueberries needed for a pint of jam -/
theorem blueberries_for_pint_of_jam :
  blueberries_per_pint 2400 6 = 200 := by
  sorry

end blueberries_for_pint_of_jam_l599_59917


namespace angle_with_complement_40percent_of_supplement_l599_59956

theorem angle_with_complement_40percent_of_supplement (x : ℝ) : 
  (90 - x = 0.4 * (180 - x)) → x = 30 := by
  sorry

end angle_with_complement_40percent_of_supplement_l599_59956


namespace baseball_average_hits_l599_59914

theorem baseball_average_hits (first_games : Nat) (first_avg : Nat) (remaining_games : Nat) (remaining_avg : Nat) : 
  first_games = 20 →
  first_avg = 2 →
  remaining_games = 10 →
  remaining_avg = 5 →
  let total_games := first_games + remaining_games
  let total_hits := first_games * first_avg + remaining_games * remaining_avg
  (total_hits : Rat) / total_games = 3 := by sorry

end baseball_average_hits_l599_59914


namespace geometric_sequence_y_value_l599_59962

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_y_value (x y z : ℝ) :
  is_geometric_sequence 1 x y z 9 → y = 3 := by
  sorry

end geometric_sequence_y_value_l599_59962


namespace problem_solution_l599_59937

theorem problem_solution (x y : ℝ) (h : 3 * x - 4 * y = 5) :
  (y = (3 * x - 5) / 4) ∧
  (y ≤ x → x ≥ -5) ∧
  (∀ a : ℝ, x + 2 * y = a ∧ x > 2 * y → a < 10) :=
by sorry

end problem_solution_l599_59937


namespace geometric_sequence_min_sum_l599_59913

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_min_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_prod : a 3 * a 5 = 64) :
  ∃ (min : ℝ), min = 16 ∧ ∀ (a' : ℕ → ℝ),
    GeometricSequence a' → (∀ n, a' n > 0) → a' 3 * a' 5 = 64 →
    a' 1 + a' 7 ≥ min :=
sorry

end geometric_sequence_min_sum_l599_59913


namespace horner_v3_value_hex_210_to_decimal_l599_59974

-- Define the polynomial and Horner's method
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def horner_step (v : ℤ) (x : ℤ) (a : ℤ) : ℤ := v * x + a

def horner_v3 (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := horner_step v0 x 5
  let v2 := horner_step v1 x 6
  horner_step v2 x 79

-- Theorem for the first part of the problem
theorem horner_v3_value : horner_v3 (-4) = -57 := by sorry

-- Define hexadecimal to decimal conversion
def hex_to_decimal (d2 d1 d0 : ℕ) : ℕ := d2 * 6^2 + d1 * 6^1 + d0 * 6^0

-- Theorem for the second part of the problem
theorem hex_210_to_decimal : hex_to_decimal 2 1 0 = 78 := by sorry

end horner_v3_value_hex_210_to_decimal_l599_59974


namespace ShortestDistance_l599_59951

/-- Line1 represents the first line (1, 2, 3) + u(1, 1, 2) -/
def Line1 (u : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => u + 1
  | 1 => u + 2
  | 2 => 2*u + 3

/-- Line2 represents the second line (2, 4, 0) + v(2, -1, 1) -/
def Line2 (v : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2*v + 2
  | 1 => -v + 4
  | 2 => v

/-- DistanceSquared calculates the squared distance between two points on the lines -/
def DistanceSquared (u v : ℝ) : ℝ :=
  (Line1 u 0 - Line2 v 0)^2 + (Line1 u 1 - Line2 v 1)^2 + (Line1 u 2 - Line2 v 2)^2

/-- ShortestDistance states that the minimum value of the square root of DistanceSquared is √5 -/
theorem ShortestDistance : 
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧ 
  ∀ (u v : ℝ), Real.sqrt (DistanceSquared u v) ≥ min_dist := by
  sorry

end ShortestDistance_l599_59951


namespace tom_last_year_games_l599_59910

/-- Represents the number of hockey games Tom attended in various scenarios -/
structure HockeyGames where
  this_year : ℕ
  missed_this_year : ℕ
  total_two_years : ℕ

/-- Calculates the number of hockey games Tom attended last year -/
def games_last_year (g : HockeyGames) : ℕ :=
  g.total_two_years - g.this_year

/-- Theorem stating that Tom attended 9 hockey games last year -/
theorem tom_last_year_games (g : HockeyGames) 
  (h1 : g.this_year = 4)
  (h2 : g.missed_this_year = 7)
  (h3 : g.total_two_years = 13) :
  games_last_year g = 9 := by
  sorry


end tom_last_year_games_l599_59910


namespace seats_formula_l599_59920

/-- The number of seats in the n-th row of a cinema -/
def seats (n : ℕ) : ℕ :=
  18 + 3 * (n - 1)

/-- Theorem: The number of seats in the n-th row is 3n + 15 -/
theorem seats_formula (n : ℕ) : seats n = 3 * n + 15 := by
  sorry

end seats_formula_l599_59920


namespace sqrt_equation_solution_l599_59927

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end sqrt_equation_solution_l599_59927


namespace max_payment_is_31_l599_59948

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := { n : ℕ // 2000 ≤ n ∧ n ≤ 2099 }

/-- Calculates the payment for a given divisor -/
def payment (d : ℕ) : ℕ :=
  match d with
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | 9 => 9
  | 11 => 11
  | _ => 0

/-- Calculates the total payment for a number based on its divisibility -/
def totalPayment (n : FourDigitNumber) : ℕ :=
  (payment 1) +
  (if n.val % 3 = 0 then payment 3 else 0) +
  (if n.val % 5 = 0 then payment 5 else 0) +
  (if n.val % 7 = 0 then payment 7 else 0) +
  (if n.val % 9 = 0 then payment 9 else 0) +
  (if n.val % 11 = 0 then payment 11 else 0)

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), totalPayment n = 31 ∧
  ∀ (m : FourDigitNumber), totalPayment m ≤ 31 :=
sorry

end max_payment_is_31_l599_59948


namespace complex_modulus_problem_l599_59940

theorem complex_modulus_problem : Complex.abs (Complex.I / (1 - Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l599_59940


namespace angle_C_indeterminate_l599_59985

/-- Represents a quadrilateral with angles A, B, C, and D -/
structure Quadrilateral where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ
  sum_360 : angleA + angleB + angleC + angleD = 360

/-- Theorem stating that ∠C cannot be determined in a quadrilateral ABCD 
    where ∠A = 80° and ∠B = 100° without information about ∠D -/
theorem angle_C_indeterminate (q : Quadrilateral) 
    (hA : q.angleA = 80) (hB : q.angleB = 100) :
  ∀ (x : ℝ), 0 < x ∧ x < 180 → 
  ∃ (q' : Quadrilateral), q'.angleA = q.angleA ∧ q'.angleB = q.angleB ∧ q'.angleC = x :=
sorry

end angle_C_indeterminate_l599_59985


namespace first_person_work_days_l599_59977

-- Define the work rates
def work_rate_prakash : ℚ := 1 / 40
def work_rate_together : ℚ := 1 / 15

-- Define the theorem
theorem first_person_work_days :
  ∃ (x : ℚ), 
    x > 0 ∧ 
    (1 / x) + work_rate_prakash = work_rate_together ∧ 
    x = 24 := by
  sorry

end first_person_work_days_l599_59977


namespace count_numbers_with_2_between_200_and_499_l599_59982

def count_numbers_with_digit_2 (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem count_numbers_with_2_between_200_and_499 :
  count_numbers_with_digit_2 200 499 = 138 := by
  sorry

end count_numbers_with_2_between_200_and_499_l599_59982


namespace orange_juice_cartons_bought_l599_59959

def prove_orange_juice_cartons : Nat :=
  let initial_money : Nat := 86
  let bread_loaves : Nat := 3
  let bread_cost : Nat := 3
  let juice_cost : Nat := 6
  let remaining_money : Nat := 59
  let spent_money : Nat := initial_money - remaining_money
  let bread_total_cost : Nat := bread_loaves * bread_cost
  let juice_total_cost : Nat := spent_money - bread_total_cost
  juice_total_cost / juice_cost

theorem orange_juice_cartons_bought :
  prove_orange_juice_cartons = 3 := by
  sorry

end orange_juice_cartons_bought_l599_59959


namespace fence_repair_boards_count_l599_59926

/-- Represents the number of boards nailed with a specific number of nails -/
structure BoardCount where
  count : ℕ
  nails_per_board : ℕ

/-- Represents a person's nailing work -/
structure NailingWork where
  first_type : BoardCount
  second_type : BoardCount

/-- Calculates the total number of nails used -/
def total_nails (work : NailingWork) : ℕ :=
  work.first_type.count * work.first_type.nails_per_board +
  work.second_type.count * work.second_type.nails_per_board

/-- Calculates the total number of boards nailed -/
def total_boards (work : NailingWork) : ℕ :=
  work.first_type.count + work.second_type.count

theorem fence_repair_boards_count :
  ∀ (petrov vasechkin : NailingWork),
    petrov.first_type.nails_per_board = 2 →
    petrov.second_type.nails_per_board = 3 →
    vasechkin.first_type.nails_per_board = 3 →
    vasechkin.second_type.nails_per_board = 5 →
    total_nails petrov = 87 →
    total_nails vasechkin = 94 →
    total_boards petrov = total_boards vasechkin →
    total_boards petrov = 30 :=
by sorry

end fence_repair_boards_count_l599_59926
