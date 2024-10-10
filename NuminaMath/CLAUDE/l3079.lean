import Mathlib

namespace matrix_inverse_proof_l3079_307910

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 5; 1, 3]

theorem matrix_inverse_proof :
  A⁻¹ = !![3, -5; -1, 2] := by
  sorry

end matrix_inverse_proof_l3079_307910


namespace sqrt_two_plus_three_times_sqrt_two_minus_three_l3079_307942

theorem sqrt_two_plus_three_times_sqrt_two_minus_three : (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = -1 := by
  sorry

end sqrt_two_plus_three_times_sqrt_two_minus_three_l3079_307942


namespace sin_cos_sum_equals_half_l3079_307945

theorem sin_cos_sum_equals_half : 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1/2 := by
  sorry

end sin_cos_sum_equals_half_l3079_307945


namespace outfit_count_l3079_307914

/-- Represents the number of shirts of each color -/
def shirts_per_color : ℕ := 4

/-- Represents the number of pants -/
def pants : ℕ := 6

/-- Represents the number of hats of each color -/
def hats_per_color : ℕ := 8

/-- Represents the number of colors -/
def colors : ℕ := 3

/-- Theorem: The number of outfits with one shirt, one pair of pants, and one hat,
    where the shirt and hat are not the same color, is 1152 -/
theorem outfit_count : 
  shirts_per_color * (colors - 1) * hats_per_color * pants = 1152 := by
  sorry


end outfit_count_l3079_307914


namespace inverse_square_theorem_l3079_307920

/-- A function representing the inverse square relationship between x and y -/
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

/-- Theorem stating the relationship between x and y -/
theorem inverse_square_theorem (k : ℝ) :
  (inverse_square_relation k 1 3) →
  (inverse_square_relation k 0.5625 4) :=
by sorry

end inverse_square_theorem_l3079_307920


namespace lg_sum_equals_two_l3079_307995

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : 2 * lg 2 + lg 25 = 2 := by sorry

end lg_sum_equals_two_l3079_307995


namespace double_elimination_tournament_players_l3079_307983

/-- Represents a double elimination tournament -/
structure DoubleEliminationTournament where
  num_players : ℕ
  num_matches : ℕ

/-- Theorem: In a double elimination tournament with 63 matches, there are 32 players -/
theorem double_elimination_tournament_players (t : DoubleEliminationTournament) 
  (h : t.num_matches = 63) : t.num_players = 32 := by
  sorry

end double_elimination_tournament_players_l3079_307983


namespace sanity_question_suffices_l3079_307917

-- Define the types of beings in Transylvania
inductive Being
| Human
| Vampire

-- Define the possible responses to the question
inductive Response
| Yes
| No

-- Define the function that represents how a being responds to the question "Are you sane?"
def respond_to_sanity_question (b : Being) : Response :=
  match b with
  | Being.Human => Response.Yes
  | Being.Vampire => Response.No

-- Define the function that determines the being type based on the response
def determine_being (r : Response) : Being :=
  match r with
  | Response.Yes => Being.Human
  | Response.No => Being.Vampire

-- Theorem: Asking "Are you sane?" is sufficient to determine if a Transylvanian is a human or a vampire
theorem sanity_question_suffices :
  ∀ (b : Being), determine_being (respond_to_sanity_question b) = b :=
by sorry


end sanity_question_suffices_l3079_307917


namespace pot_count_l3079_307990

/-- The number of pots given the number of flowers and sticks per pot and the total number of flowers and sticks -/
def number_of_pots (flowers_per_pot : ℕ) (sticks_per_pot : ℕ) (total_items : ℕ) : ℕ :=
  total_items / (flowers_per_pot + sticks_per_pot)

/-- Theorem stating that there are 466 pots given the conditions -/
theorem pot_count : number_of_pots 53 181 109044 = 466 := by
  sorry

#eval number_of_pots 53 181 109044

end pot_count_l3079_307990


namespace carla_class_size_l3079_307974

theorem carla_class_size :
  let students_in_restroom : ℕ := 2
  let absent_students : ℕ := 3 * students_in_restroom - 1
  let total_desks : ℕ := 4 * 6
  let occupied_desks : ℕ := (2 * total_desks) / 3
  let students_present : ℕ := occupied_desks
  students_in_restroom + absent_students + students_present = 23 := by
sorry

end carla_class_size_l3079_307974


namespace sum_of_first_40_digits_of_fraction_l3079_307935

-- Define the fraction
def fraction : ℚ := 1 / 1234

-- Define a function to get the nth digit after the decimal point
def nthDigitAfterDecimal (n : ℕ) : ℕ := sorry

-- Define the sum of the first 40 digits after the decimal point
def sumOfFirst40Digits : ℕ := (List.range 40).map nthDigitAfterDecimal |>.sum

-- Theorem statement
theorem sum_of_first_40_digits_of_fraction :
  sumOfFirst40Digits = 218 := by sorry

end sum_of_first_40_digits_of_fraction_l3079_307935


namespace x_varies_as_four_thirds_power_of_z_l3079_307987

-- Define the variables
variable (x y z : ℝ)
-- Define constants of proportionality
variable (k j : ℝ)

-- Define the relationships
def x_varies_as_y_squared : Prop := ∃ k > 0, x = k * y^2
def y_varies_as_cube_root_z_squared : Prop := ∃ j > 0, y = j * (z^2)^(1/3)

-- State the theorem
theorem x_varies_as_four_thirds_power_of_z 
  (h1 : x_varies_as_y_squared x y) 
  (h2 : y_varies_as_cube_root_z_squared y z) : 
  ∃ m > 0, x = m * z^(4/3) := by
  sorry

end x_varies_as_four_thirds_power_of_z_l3079_307987


namespace sofa_love_seat_cost_l3079_307905

/-- The cost of a love seat and sofa, where the sofa costs double the love seat -/
def total_cost (love_seat_cost : ℝ) : ℝ :=
  love_seat_cost + 2 * love_seat_cost

/-- Theorem stating that the total cost is $444 when the love seat costs $148 -/
theorem sofa_love_seat_cost : total_cost 148 = 444 := by
  sorry

end sofa_love_seat_cost_l3079_307905


namespace soap_cost_l3079_307994

/-- The cost of a bar of soap given monthly usage and two-year expenditure -/
theorem soap_cost (monthly_usage : ℕ) (two_year_expenditure : ℚ) :
  monthly_usage = 1 →
  two_year_expenditure = 96 →
  two_year_expenditure / (24 : ℚ) = 4 := by
sorry

end soap_cost_l3079_307994


namespace extended_pattern_ratio_l3079_307980

def original_width : ℕ := 5
def original_height : ℕ := 6
def original_black_tiles : ℕ := 12
def original_white_tiles : ℕ := 18
def border_width : ℕ := 1

def extended_width : ℕ := original_width + 2 * border_width
def extended_height : ℕ := original_height + 2 * border_width

def total_extended_tiles : ℕ := extended_width * extended_height
def new_white_tiles : ℕ := total_extended_tiles - (original_width * original_height)
def total_white_tiles : ℕ := original_white_tiles + new_white_tiles

theorem extended_pattern_ratio :
  (original_black_tiles : ℚ) / total_white_tiles = 3 / 11 := by
  sorry

end extended_pattern_ratio_l3079_307980


namespace suspension_ratio_l3079_307912

/-- The number of fingers and toes a typical person has -/
def typical_fingers_and_toes : ℕ := 20

/-- The number of days Kris is suspended for each bullying instance -/
def suspension_days_per_instance : ℕ := 3

/-- The number of bullying instances Kris is responsible for -/
def bullying_instances : ℕ := 20

/-- Kris's total suspension days -/
def total_suspension_days : ℕ := suspension_days_per_instance * bullying_instances

theorem suspension_ratio :
  total_suspension_days / typical_fingers_and_toes = 3 :=
by sorry

end suspension_ratio_l3079_307912


namespace beef_cost_calculation_l3079_307961

/-- Proves that the cost of a pound of beef is $5 given the initial amount,
    cheese cost, quantities purchased, and remaining amount. -/
theorem beef_cost_calculation (initial_amount : ℕ) (cheese_cost : ℕ) 
  (cheese_quantity : ℕ) (beef_quantity : ℕ) (remaining_amount : ℕ) :
  initial_amount = 87 →
  cheese_cost = 7 →
  cheese_quantity = 3 →
  beef_quantity = 1 →
  remaining_amount = 61 →
  initial_amount - remaining_amount - (cheese_cost * cheese_quantity) = 5 :=
by sorry

end beef_cost_calculation_l3079_307961


namespace seating_arrangements_l3079_307991

/-- The number of seats on the bench -/
def total_seats : ℕ := 7

/-- The number of people to be seated -/
def people_to_seat : ℕ := 4

/-- The number of empty seats -/
def empty_seats : ℕ := total_seats - people_to_seat

/-- The total number of unrestricted seating arrangements -/
def total_arrangements : ℕ := 840

theorem seating_arrangements :
  (∃ (arrangements_with_adjacent : ℕ),
    arrangements_with_adjacent = total_arrangements - 24 ∧
    arrangements_with_adjacent = 816) ∧
  (∃ (arrangements_without_all_empty_adjacent : ℕ),
    arrangements_without_all_empty_adjacent = total_arrangements - 120 ∧
    arrangements_without_all_empty_adjacent = 720) := by
  sorry

end seating_arrangements_l3079_307991


namespace vector_collinearity_l3079_307988

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

/-- Collinearity of two vectors in ℝ² -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- The main theorem -/
theorem vector_collinearity (m : ℝ) :
  collinear ((m * a.1 + 4 * b.1, m * a.2 + 4 * b.2)) (a.1 - 2 * b.1, a.2 - 2 * b.2) →
  m = -2 := by
  sorry

end vector_collinearity_l3079_307988


namespace quadratic_vertex_form_l3079_307908

theorem quadratic_vertex_form (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
sorry

end quadratic_vertex_form_l3079_307908


namespace boys_joined_school_l3079_307957

theorem boys_joined_school (initial_boys final_boys : ℕ) 
  (h1 : initial_boys = 214)
  (h2 : final_boys = 1124) :
  final_boys - initial_boys = 910 := by
  sorry

end boys_joined_school_l3079_307957


namespace badminton_match_31_probability_l3079_307927

def badminton_match_probability (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p)

theorem badminton_match_31_probability :
  badminton_match_probability (2/3) = 8/27 := by
  sorry

end badminton_match_31_probability_l3079_307927


namespace fourth_month_sale_problem_l3079_307941

/-- Calculates the sale in the fourth month given the sales of other months and the average --/
def fourthMonthSale (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem stating the sale in the fourth month given the problem conditions --/
theorem fourth_month_sale_problem :
  fourthMonthSale 5420 5660 6200 6500 8270 6400 = 6350 := by
  sorry

#eval fourthMonthSale 5420 5660 6200 6500 8270 6400

end fourth_month_sale_problem_l3079_307941


namespace matchboxes_per_box_l3079_307900

/-- Proves that the number of matchboxes in each box is 20, given the total number of boxes,
    sticks per matchbox, and total number of sticks. -/
theorem matchboxes_per_box 
  (total_boxes : ℕ) 
  (sticks_per_matchbox : ℕ) 
  (total_sticks : ℕ) 
  (h1 : total_boxes = 4)
  (h2 : sticks_per_matchbox = 300)
  (h3 : total_sticks = 24000) :
  total_sticks / sticks_per_matchbox / total_boxes = 20 := by
  sorry

#eval 24000 / 300 / 4  -- Should output 20

end matchboxes_per_box_l3079_307900


namespace yoongi_score_l3079_307907

theorem yoongi_score (yoongi eunji yuna : ℕ) 
  (h1 : eunji = yoongi - 25)
  (h2 : yuna = eunji - 20)
  (h3 : yuna = 8) :
  yoongi = 53 := by
  sorry

end yoongi_score_l3079_307907


namespace x_plus_y_values_l3079_307937

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 5) (h2 : y = Real.sqrt 9) :
  x + y = -2 ∨ x + y = 8 := by
  sorry

end x_plus_y_values_l3079_307937


namespace matrix_cube_eq_matrix_plus_identity_det_positive_l3079_307993

open Matrix

theorem matrix_cube_eq_matrix_plus_identity_det_positive :
  ∀ (n : ℕ), ∃ (A : Matrix (Fin n) (Fin n) ℝ), A ^ 3 = A + 1 →
  ∀ (A : Matrix (Fin n) (Fin n) ℝ), A ^ 3 = A + 1 → 0 < det A :=
by sorry

end matrix_cube_eq_matrix_plus_identity_det_positive_l3079_307993


namespace function_identity_l3079_307902

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, 2 * f x - f (-x) = 3 * x) :
  ∀ x, f x = x := by
  sorry

end function_identity_l3079_307902


namespace calculation_result_l3079_307919

theorem calculation_result (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : m = -2)  -- m is a negative number with an absolute value of 2
  : m + c * d + a + b + (c * d) ^ 2010 = 0 := by
  sorry

end calculation_result_l3079_307919


namespace lower_bound_of_expression_l3079_307969

theorem lower_bound_of_expression (L : ℤ) : 
  (∃ (S : Finset ℤ), 
    (∀ n ∈ S, L < 4*n + 7 ∧ 4*n + 7 < 120) ∧ 
    S.card = 30) →
  L = 5 :=
sorry

end lower_bound_of_expression_l3079_307969


namespace jackson_flight_distance_l3079_307954

theorem jackson_flight_distance (beka_miles jackson_miles : ℕ) 
  (h1 : beka_miles = 873)
  (h2 : beka_miles = jackson_miles + 310) : 
  jackson_miles = 563 := by
sorry

end jackson_flight_distance_l3079_307954


namespace adjacent_complementary_angles_are_complementary_l3079_307992

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- Two angles are adjacent if they share a common vertex and a common side,
    but do not overlap -/
def Adjacent (α β : ℝ) : Prop := True  -- We simplify this for the purpose of the statement

theorem adjacent_complementary_angles_are_complementary
  (α β : ℝ) (h1 : Adjacent α β) (h2 : Complementary α β) :
  Complementary α β :=
sorry

end adjacent_complementary_angles_are_complementary_l3079_307992


namespace angelina_walking_speed_l3079_307949

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_to_grocery : ℝ) 
  (distance_grocery_to_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_home_to_grocery = 100)
  (h2 : distance_grocery_to_gym = 180)
  (h3 : time_difference = 40)
  : ∃ (v : ℝ), 
    (distance_home_to_grocery / v - distance_grocery_to_gym / (2 * v) = time_difference) ∧ 
    (2 * v = 1/2) :=
by sorry

end angelina_walking_speed_l3079_307949


namespace counterclockwise_notation_l3079_307962

/-- Represents the direction of rotation -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Represents a rotation with a direction and an angle -/
structure Rotation :=
  (direction : RotationDirection)
  (angle : ℝ)

/-- Notation for a rotation -/
def rotationNotation (r : Rotation) : ℝ :=
  match r.direction with
  | RotationDirection.Clockwise => r.angle
  | RotationDirection.Counterclockwise => -r.angle

theorem counterclockwise_notation 
  (h : rotationNotation { direction := RotationDirection.Clockwise, angle := 60 } = 60) :
  rotationNotation { direction := RotationDirection.Counterclockwise, angle := 15 } = -15 :=
by
  sorry

end counterclockwise_notation_l3079_307962


namespace short_pencil_cost_proof_l3079_307929

/-- The cost of a short pencil in dollars -/
def short_pencil_cost : ℚ := 0.4

/-- The cost of a pencil with eraser in dollars -/
def eraser_pencil_cost : ℚ := 0.8

/-- The cost of a regular pencil in dollars -/
def regular_pencil_cost : ℚ := 0.5

/-- The number of pencils with eraser sold -/
def eraser_pencils_sold : ℕ := 200

/-- The number of regular pencils sold -/
def regular_pencils_sold : ℕ := 40

/-- The number of short pencils sold -/
def short_pencils_sold : ℕ := 35

/-- The total revenue from all sales in dollars -/
def total_revenue : ℚ := 194

theorem short_pencil_cost_proof :
  short_pencil_cost * short_pencils_sold +
  eraser_pencil_cost * eraser_pencils_sold +
  regular_pencil_cost * regular_pencils_sold = total_revenue :=
by sorry

end short_pencil_cost_proof_l3079_307929


namespace sum_of_roots_equals_six_l3079_307924

theorem sum_of_roots_equals_six : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16
  ∃ r₁ r₂ : ℝ, (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ + r₂ = 6 := by
sorry

end sum_of_roots_equals_six_l3079_307924


namespace carrots_per_pound_l3079_307948

/-- Given the number of carrots in three beds and the total weight of the harvest,
    calculate the number of carrots that weigh one pound. -/
theorem carrots_per_pound 
  (bed1 bed2 bed3 : ℕ) 
  (total_weight : ℕ) 
  (h1 : bed1 = 55)
  (h2 : bed2 = 101)
  (h3 : bed3 = 78)
  (h4 : total_weight = 39) :
  (bed1 + bed2 + bed3) / total_weight = 6 := by
  sorry

#check carrots_per_pound

end carrots_per_pound_l3079_307948


namespace arithmetic_sum_10_l3079_307981

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) : ℕ := n * (2 * n + 1)

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence with general term a_n = 2n + 1 is 120 -/
theorem arithmetic_sum_10 : arithmetic_sum 10 = 120 := by
  sorry

end arithmetic_sum_10_l3079_307981


namespace solve_equation_l3079_307984

theorem solve_equation : ∃ x : ℝ, 90 + (x * 12) / (180 / 3) = 91 ∧ x = 5 := by
  sorry

end solve_equation_l3079_307984


namespace power_mod_prime_l3079_307923

theorem power_mod_prime (p : Nat) (h : p.Prime) :
  (3 : ZMod p)^2020 = 8 :=
by
  sorry

end power_mod_prime_l3079_307923


namespace value_of_M_l3079_307976

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.55 * 4500) ∧ (M = 9900) := by
  sorry

end value_of_M_l3079_307976


namespace max_value_inequality_l3079_307903

theorem max_value_inequality (A : ℝ) (h : A > 0) :
  let M := max (2 + A / 2) (2 * Real.sqrt A)
  ∀ x y : ℝ, x > 0 → y > 0 →
    1 / x + 1 / y + A / (x + y) ≥ M / Real.sqrt (x * y) :=
by sorry

end max_value_inequality_l3079_307903


namespace right_triangle_sine_cosine_l3079_307934

theorem right_triangle_sine_cosine (D E F : ℝ) : 
  E = 90 → -- angle E is 90 degrees
  3 * Real.sin D = 4 * Real.cos D → -- given condition
  Real.sin D = 4/5 := by sorry

end right_triangle_sine_cosine_l3079_307934


namespace baker_cakes_l3079_307985

/-- The initial number of cakes Baker made -/
def initial_cakes : ℕ := 169

/-- The number of cakes Baker's friend bought -/
def bought_cakes : ℕ := 137

/-- The number of cakes Baker has left -/
def remaining_cakes : ℕ := 32

/-- Theorem stating that the initial number of cakes is equal to the sum of bought cakes and remaining cakes -/
theorem baker_cakes : initial_cakes = bought_cakes + remaining_cakes := by
  sorry

end baker_cakes_l3079_307985


namespace remainder_problem_l3079_307971

theorem remainder_problem (s t u : ℕ) 
  (hs : s % 12 = 4)
  (ht : t % 12 = 5)
  (hu : u % 12 = 7)
  (hst : s > t)
  (htu : t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end remainder_problem_l3079_307971


namespace inequality_proof_l3079_307936

theorem inequality_proof (n : ℕ) (hn : n > 1) :
  1 / Real.exp 1 - 1 / (n * Real.exp 1) < (1 - 1 / n : ℝ)^n ∧
  (1 - 1 / n : ℝ)^n < 1 / Real.exp 1 - 1 / (2 * n * Real.exp 1) :=
by sorry

end inequality_proof_l3079_307936


namespace lion_weight_is_41_3_l3079_307998

/-- The weight of a lion in kilograms -/
def lion_weight : ℝ := 41.3

/-- The weight of a tiger in kilograms -/
def tiger_weight : ℝ := lion_weight - 4.8

/-- The weight of a panda in kilograms -/
def panda_weight : ℝ := tiger_weight - 7.7

/-- Theorem stating that the weight of a lion is 41.3 kg given the conditions -/
theorem lion_weight_is_41_3 : 
  lion_weight = 41.3 ∧ 
  tiger_weight = lion_weight - 4.8 ∧
  panda_weight = tiger_weight - 7.7 ∧
  lion_weight + tiger_weight + panda_weight = 106.6 := by
  sorry

#check lion_weight_is_41_3

end lion_weight_is_41_3_l3079_307998


namespace magical_stack_size_l3079_307966

/-- A stack of cards is magical if it satisfies certain conditions -/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (card_197_position : ℕ)
  (card_197_retains_position : card_197_position = 197)
  (is_magical : ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ total_cards)

/-- The number of cards in a magical stack where card 197 retains its position is 590 -/
theorem magical_stack_size (stack : MagicalStack) : stack.total_cards = 590 :=
by sorry

end magical_stack_size_l3079_307966


namespace power_function_through_point_l3079_307955

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 2 = 4 →
  f 9 = 81 := by
sorry

end power_function_through_point_l3079_307955


namespace football_games_indeterminate_l3079_307906

theorem football_games_indeterminate 
  (night_games : ℕ) 
  (keith_missed : ℕ) 
  (keith_attended : ℕ) 
  (h1 : night_games = 4) 
  (h2 : keith_missed = 4) 
  (h3 : keith_attended = 4) :
  ¬ ∃ (total_games : ℕ), 
    (total_games ≥ night_games) ∧ 
    (total_games = keith_missed + keith_attended) :=
by sorry

end football_games_indeterminate_l3079_307906


namespace motorboat_stream_speed_l3079_307960

/-- Proves that the speed of the stream is 3 kmph given the conditions of the motorboat problem -/
theorem motorboat_stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 21) 
  (h2 : distance = 72) 
  (h3 : total_time = 7) :
  ∃ (stream_speed : ℝ), 
    stream_speed = 3 ∧ 
    distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time :=
by
  sorry

end motorboat_stream_speed_l3079_307960


namespace sharons_salary_increase_l3079_307967

theorem sharons_salary_increase (S : ℝ) (x : ℝ) : 
  S * 1.08 = 324 → S * (1 + x / 100) = 330 → x = 10 := by
  sorry

end sharons_salary_increase_l3079_307967


namespace functional_equation_solution_l3079_307950

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2) + f (x * y) = f x * f y + y * f x + x * f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    is either constantly zero or the negation function. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = -x) := by
  sorry


end functional_equation_solution_l3079_307950


namespace general_solution_second_order_recurrence_l3079_307943

/-- Second-order linear recurrence sequence -/
def RecurrenceSequence (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 2) = a * u (n + 1) + b * u n

/-- Characteristic polynomial of the recurrence sequence -/
def CharacteristicPolynomial (a b : ℝ) (X : ℝ) : ℝ :=
  X^2 - a*X - b

theorem general_solution_second_order_recurrence
  (a b : ℝ) (u : ℕ → ℝ) (r₁ r₂ : ℝ) :
  RecurrenceSequence a b u →
  r₁ ≠ r₂ →
  CharacteristicPolynomial a b r₁ = 0 →
  CharacteristicPolynomial a b r₂ = 0 →
  ∃ c d : ℝ, ∀ n, u n = c * r₁^n + d * r₂^n ∧
    c = (u 1 - u 0 * r₂) / (r₁ - r₂) ∧
    d = (u 0 * r₁ - u 1) / (r₁ - r₂) :=
sorry

end general_solution_second_order_recurrence_l3079_307943


namespace measure_eight_liters_possible_l3079_307921

/-- Represents the state of the buckets -/
structure BucketState where
  b10 : ℕ  -- Amount of water in the 10-liter bucket
  b6 : ℕ   -- Amount of water in the 6-liter bucket

/-- Represents a single operation on the buckets -/
inductive BucketOperation
  | FillFromRiver (bucket : ℕ)  -- Fill a bucket from the river
  | EmptyToRiver (bucket : ℕ)   -- Empty a bucket to the river
  | PourBetweenBuckets          -- Pour from one bucket to another

/-- Applies a single operation to the current state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  sorry

/-- Checks if the given sequence of operations results in 8 liters in one bucket -/
def isValidSolution (operations : List BucketOperation) : Bool :=
  sorry

/-- Theorem stating that it's possible to measure 8 liters using the given buckets -/
theorem measure_eight_liters_possible :
  ∃ (operations : List BucketOperation), isValidSolution operations :=
  sorry

end measure_eight_liters_possible_l3079_307921


namespace ball_arrangements_count_l3079_307953

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements of placing 5 numbered balls into 3 boxes,
    where two boxes contain 2 balls each and one box contains 1 ball --/
def ball_arrangements : ℕ :=
  choose 3 2 * choose 5 2 * choose 3 2

theorem ball_arrangements_count : ball_arrangements = 90 := by sorry

end ball_arrangements_count_l3079_307953


namespace simplify_expression_l3079_307978

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 9*y = 45*x + 9*y := by
  sorry

end simplify_expression_l3079_307978


namespace parking_problem_l3079_307997

/-- Represents the number of parking spaces -/
def total_spaces : ℕ := 7

/-- Represents the number of cars -/
def num_cars : ℕ := 3

/-- Represents the number of consecutive empty spaces -/
def empty_spaces : ℕ := 4

/-- Represents the total number of units to arrange (cars + empty space block) -/
def total_units : ℕ := num_cars + 1

/-- The number of different parking arrangements -/
def parking_arrangements : ℕ := Nat.factorial total_units

theorem parking_problem :
  parking_arrangements = 24 :=
sorry

end parking_problem_l3079_307997


namespace fred_balloons_l3079_307989

theorem fred_balloons (initial : ℕ) (to_sandy : ℕ) (to_bob : ℕ) :
  initial = 709 →
  to_sandy = 221 →
  to_bob = 153 →
  initial - to_sandy - to_bob = 335 := by
  sorry

end fred_balloons_l3079_307989


namespace intersection_m_value_l3079_307951

theorem intersection_m_value (x y : ℝ) (m : ℝ) : 
  (3 * x + y = m) →
  (-0.75 * x + y = -22) →
  (x = 6) →
  (m = 0.5) := by
  sorry

end intersection_m_value_l3079_307951


namespace exterior_angle_HGI_exterior_angle_is_81_degrees_l3079_307972

-- Define the polygons
def Octagon : Type := Unit
def Decagon : Type := Unit

-- Define the properties of the polygons
axiom is_regular_octagon : Octagon → Prop
axiom is_regular_decagon : Decagon → Prop

-- Define the interior angles
def interior_angle_octagon (o : Octagon) (h : is_regular_octagon o) : ℝ := 135
def interior_angle_decagon (d : Decagon) (h : is_regular_decagon d) : ℝ := 144

-- Define the configuration
structure Configuration :=
  (o : Octagon)
  (d : Decagon)
  (ho : is_regular_octagon o)
  (hd : is_regular_decagon d)
  (share_side : Prop)

-- State the theorem
theorem exterior_angle_HGI (c : Configuration) : ℝ :=
  360 - interior_angle_octagon c.o c.ho - interior_angle_decagon c.d c.hd

-- The main theorem to prove
theorem exterior_angle_is_81_degrees (c : Configuration) :
  exterior_angle_HGI c = 81 := by sorry

end exterior_angle_HGI_exterior_angle_is_81_degrees_l3079_307972


namespace pigeon_problem_l3079_307913

theorem pigeon_problem (x y : ℕ) : 
  (y + 1 = (1/6) * (x + y + 1)) → 
  (x - 1 = y + 1) → 
  (x = 4 ∧ y = 2) :=
by sorry

end pigeon_problem_l3079_307913


namespace inequality_solution_l3079_307946

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 3/4) ↔ (-2 < x ∧ x ≤ 2) :=
by sorry

end inequality_solution_l3079_307946


namespace sum_21_terms_arithmetic_sequence_l3079_307964

/-- Arithmetic sequence with first term 3 and common difference 10 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  3 + (n - 1) * 10

/-- Sum of the first n terms of the arithmetic sequence -/
def sumArithmeticSequence (n : ℕ) : ℤ :=
  n * (3 + arithmeticSequence n) / 2

theorem sum_21_terms_arithmetic_sequence :
  sumArithmeticSequence 21 = 2163 := by
  sorry

end sum_21_terms_arithmetic_sequence_l3079_307964


namespace toothpick_structure_count_l3079_307922

/-- Calculates the number of toothpicks in a rectangular grid --/
def rectangle_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Calculates the number of toothpicks in a right-angled triangle --/
def triangle_toothpicks (base : ℕ) : ℕ :=
  base + (Int.sqrt (2 * base * base)).toNat + 1

/-- The total number of toothpicks in the structure --/
def total_toothpicks (length width : ℕ) : ℕ :=
  rectangle_toothpicks length width + triangle_toothpicks width

theorem toothpick_structure_count :
  total_toothpicks 40 20 = 1709 := by
  sorry

end toothpick_structure_count_l3079_307922


namespace frog_count_l3079_307965

theorem frog_count (total_eyes : ℕ) (eyes_per_frog : ℕ) (h1 : total_eyes > 0) (h2 : eyes_per_frog > 0) :
  total_eyes / eyes_per_frog = 4 →
  total_eyes = 8 ∧ eyes_per_frog = 2 :=
by sorry

end frog_count_l3079_307965


namespace right_triangle_inequality_l3079_307939

theorem right_triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 = b^2 + c^2)
  (h_order : a > b ∧ b > c)
  (h_n : n > 2) : 
  a^n > b^n + c^n := by
sorry

end right_triangle_inequality_l3079_307939


namespace vertical_distance_theorem_l3079_307933

def f (x : ℝ) := |x|
def g (x : ℝ) := -x^2 - 4*x - 3

def solution_set : Set ℝ := {(-5 + Real.sqrt 29)/2, (-5 - Real.sqrt 29)/2, (-3 + Real.sqrt 13)/2, (-3 - Real.sqrt 13)/2}

theorem vertical_distance_theorem :
  ∀ x : ℝ, (f x - g x = 4 ∨ g x - f x = 4) ↔ x ∈ solution_set := by sorry

end vertical_distance_theorem_l3079_307933


namespace final_book_count_l3079_307928

/-- Represents the number of books in each genre -/
structure BookCollection :=
  (novels : ℕ)
  (science : ℕ)
  (cookbooks : ℕ)
  (philosophy : ℕ)
  (history : ℕ)
  (selfHelp : ℕ)

/-- Represents the donation percentages for each genre -/
structure DonationPercentages :=
  (novels : ℚ)
  (science : ℚ)
  (cookbooks : ℚ)
  (philosophy : ℚ)
  (history : ℚ)
  (selfHelp : ℚ)

def initialCollection : BookCollection :=
  { novels := 75
  , science := 55
  , cookbooks := 40
  , philosophy := 35
  , history := 25
  , selfHelp := 20 }

def donationPercentages : DonationPercentages :=
  { novels := 3/5
  , science := 3/4
  , cookbooks := 1/2
  , philosophy := 3/10
  , history := 1/4
  , selfHelp := 1 }

def recyclePercentage : ℚ := 1/20

def newBooksAcquired : ℕ := 24

theorem final_book_count
  (total : ℕ)
  (h1 : total = initialCollection.novels + initialCollection.science +
                initialCollection.cookbooks + initialCollection.philosophy +
                initialCollection.history + initialCollection.selfHelp)
  (h2 : total = 250) :
  ∃ (donatedBooks recycledBooks remainingBooks : ℕ),
    donatedBooks = ⌊initialCollection.novels * donationPercentages.novels⌋ +
                   ⌊initialCollection.science * donationPercentages.science⌋ +
                   ⌊initialCollection.cookbooks * donationPercentages.cookbooks⌋ +
                   ⌊initialCollection.philosophy * donationPercentages.philosophy⌋ +
                   ⌊initialCollection.history * donationPercentages.history⌋ +
                   ⌊initialCollection.selfHelp * donationPercentages.selfHelp⌋ ∧
    recycledBooks = ⌊(donatedBooks : ℚ) * recyclePercentage⌋ ∧
    remainingBooks = total - donatedBooks + recycledBooks ∧
    remainingBooks + newBooksAcquired = 139 :=
by sorry

end final_book_count_l3079_307928


namespace room_length_calculation_l3079_307947

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 10 → width = 2 → area = length * width → length = 5 := by
sorry

end room_length_calculation_l3079_307947


namespace sidney_wednesday_jumping_jacks_l3079_307956

/-- The number of jumping jacks Sidney did on Wednesday -/
def sidney_wednesday : ℕ := sorry

/-- The total number of jumping jacks Sidney did -/
def sidney_total : ℕ := sorry

/-- The number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

theorem sidney_wednesday_jumping_jacks :
  sidney_wednesday = 40 ∧
  sidney_total = sidney_wednesday + 106 ∧
  brooke_total = 3 * sidney_total :=
by sorry

end sidney_wednesday_jumping_jacks_l3079_307956


namespace paint_usage_l3079_307915

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1 / 9)
  (h3 : second_week_fraction = 1 / 5) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  let total_usage := first_week_usage + second_week_usage
  total_usage = 104 := by
sorry

end paint_usage_l3079_307915


namespace scientific_notation_of_1650000000_l3079_307926

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original number to be expressed in scientific notation -/
def originalNumber : ℝ := 1650000000

/-- The number of significant figures to keep -/
def sigFigs : ℕ := 3

theorem scientific_notation_of_1650000000 :
  toScientificNotation originalNumber sigFigs =
    ScientificNotation.mk 1.65 9 (by norm_num) :=
  sorry

end scientific_notation_of_1650000000_l3079_307926


namespace negation_of_proposition_l3079_307979

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 2 → x > 3)) ↔ (∃ x : ℝ, x > 2 ∧ x ≤ 3) := by
  sorry

end negation_of_proposition_l3079_307979


namespace adjacent_rectangle_area_l3079_307925

-- Define the structure of our rectangle
structure DividedRectangle where
  total_length : ℝ
  total_width : ℝ
  length_split : ℝ -- Point where length is split
  width_split : ℝ -- Point where width is split
  inner_split : ℝ -- Point where the largest rectangle is split

-- Define our specific rectangle
def our_rectangle : DividedRectangle where
  total_length := 5
  total_width := 13
  length_split := 3
  width_split := 9
  inner_split := 4

-- Define areas of known rectangles
def area1 : ℝ := 12
def area2 : ℝ := 15
def area3 : ℝ := 20
def area4 : ℝ := 18
def inner_area : ℝ := 8

-- Theorem to prove
theorem adjacent_rectangle_area (r : DividedRectangle) :
  r.length_split * r.inner_split = area3 - inner_area ∧
  (r.total_length - r.length_split) * r.inner_split = inner_area ∧
  (r.total_length - r.length_split) * (r.total_width - r.width_split) = area4 →
  area4 = 18 := by sorry

end adjacent_rectangle_area_l3079_307925


namespace book_profit_calculation_l3079_307932

/-- Calculate the overall percent profit for two books with given costs, markups, and discounts -/
theorem book_profit_calculation (cost_a cost_b : ℝ) (markup_a markup_b : ℝ) (discount_a discount_b : ℝ) :
  cost_a = 50 →
  cost_b = 70 →
  markup_a = 0.4 →
  markup_b = 0.6 →
  discount_a = 0.15 →
  discount_b = 0.2 →
  let marked_price_a := cost_a * (1 + markup_a)
  let marked_price_b := cost_b * (1 + markup_b)
  let sale_price_a := marked_price_a * (1 - discount_a)
  let sale_price_b := marked_price_b * (1 - discount_b)
  let total_cost := cost_a + cost_b
  let total_sale_price := sale_price_a + sale_price_b
  let total_profit := total_sale_price - total_cost
  let percent_profit := (total_profit / total_cost) * 100
  percent_profit = 24.25 := by sorry

end book_profit_calculation_l3079_307932


namespace marble_distribution_correct_l3079_307977

/-- Represents the distribution of marbles among four boys -/
structure MarbleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The rule for distributing marbles based on a parameter x -/
def distributionRule (x : ℕ) : MarbleDistribution :=
  { first := 3 * x + 2
  , second := x + 1
  , third := 2 * x - 1
  , fourth := x }

/-- Theorem stating that the given distribution satisfies the problem conditions -/
theorem marble_distribution_correct : ∃ x : ℕ, 
  let d := distributionRule x
  d.first = 22 ∧
  d.second = 8 ∧
  d.third = 12 ∧
  d.fourth = 7 ∧
  d.first + d.second + d.third + d.fourth = 49 := by
  sorry

end marble_distribution_correct_l3079_307977


namespace pete_reads_300_books_l3079_307968

/-- The number of books Pete reads across two years given the conditions -/
def petes_total_books (matts_second_year_books : ℕ) : ℕ :=
  let matts_first_year_books := matts_second_year_books * 2 / 3
  let petes_first_year_books := matts_first_year_books * 2
  let petes_second_year_books := petes_first_year_books * 2
  petes_first_year_books + petes_second_year_books

/-- Theorem stating that Pete reads 300 books across both years -/
theorem pete_reads_300_books : petes_total_books 75 = 300 := by
  sorry

end pete_reads_300_books_l3079_307968


namespace tyson_race_time_l3079_307909

/-- Calculates the total time Tyson spent in his races given his swimming speeds and race conditions. -/
theorem tyson_race_time (lake_speed ocean_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) : 
  lake_speed = 3 → 
  ocean_speed = 2.5 → 
  total_races = 10 → 
  race_distance = 3 → 
  (total_races / 2 : ℝ) * (race_distance / lake_speed) + 
  (total_races / 2 : ℝ) * (race_distance / ocean_speed) = 11 := by
  sorry

#check tyson_race_time

end tyson_race_time_l3079_307909


namespace alice_departure_time_l3079_307901

/-- Proof that Alice must leave 30 minutes after Bob to arrive in city B just before him. -/
theorem alice_departure_time (distance : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) 
  (h1 : distance = 220)
  (h2 : bob_speed = 40)
  (h3 : alice_speed = 44) :
  (distance / bob_speed - distance / alice_speed) * 60 = 30 := by
  sorry

#check alice_departure_time

end alice_departure_time_l3079_307901


namespace max_t_is_e_l3079_307904

theorem max_t_is_e (t : ℝ) : 
  (∀ a b : ℝ, 0 < a → a < b → b < t → b * Real.log a < a * Real.log b) →
  t ≤ Real.exp 1 :=
sorry

end max_t_is_e_l3079_307904


namespace thirty_sided_polygon_diagonals_l3079_307973

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end thirty_sided_polygon_diagonals_l3079_307973


namespace candy_distribution_l3079_307952

theorem candy_distribution (S M L : ℕ) 
  (total : S + M + L = 110)
  (without_jelly : S + L = 100)
  (relation : M + L = S + M + 20) :
  S = 40 ∧ L = 60 ∧ M = 10 := by
  sorry

end candy_distribution_l3079_307952


namespace fraction_equality_l3079_307931

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_equality_l3079_307931


namespace wire_length_proof_l3079_307996

-- Define the area of the square field
def field_area : ℝ := 69696

-- Define the number of times the wire goes around the field
def rounds : ℕ := 15

-- Theorem statement
theorem wire_length_proof :
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  rounds * perimeter = 15840 := by
  sorry

end wire_length_proof_l3079_307996


namespace carol_invitations_proof_l3079_307918

/-- The number of invitations Carol is sending out -/
def total_invitations : ℕ := 12

/-- The number of packs Carol bought -/
def number_of_packs : ℕ := 3

/-- The number of invitations in each pack -/
def invitations_per_pack : ℕ := total_invitations / number_of_packs

theorem carol_invitations_proof :
  invitations_per_pack = 4 ∧
  total_invitations = number_of_packs * invitations_per_pack :=
by sorry

end carol_invitations_proof_l3079_307918


namespace max_angle_A1MC1_is_pi_over_2_l3079_307986

/-- Represents a right square prism -/
structure RightSquarePrism where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- Represents a point on an edge of the prism -/
structure EdgePoint where
  x : ℝ
  valid : 0 ≤ x ∧ x ≤ 1

/-- Calculates the angle A₁MC₁ given a point M on edge AB -/
def angle_A1MC1 (prism : RightSquarePrism) (M : EdgePoint) : ℝ := sorry

/-- Theorem: The maximum value of angle A₁MC₁ in a right square prism is π/2 -/
theorem max_angle_A1MC1_is_pi_over_2 (prism : RightSquarePrism) :
  ∃ M : EdgePoint, ∀ N : EdgePoint, angle_A1MC1 prism M ≥ angle_A1MC1 prism N ∧ 
  angle_A1MC1 prism M = π / 2 :=
sorry

end max_angle_A1MC1_is_pi_over_2_l3079_307986


namespace smallest_valid_survey_size_l3079_307958

def is_valid_survey_size (N : ℕ) : Prop :=
  (N * 1 / 10 : ℚ).num % (N * 1 / 10 : ℚ).den = 0 ∧
  (N * 3 / 10 : ℚ).num % (N * 3 / 10 : ℚ).den = 0 ∧
  (N * 2 / 5 : ℚ).num % (N * 2 / 5 : ℚ).den = 0

theorem smallest_valid_survey_size :
  ∃ (N : ℕ), N > 0 ∧ is_valid_survey_size N ∧ ∀ (M : ℕ), M > 0 ∧ is_valid_survey_size M → N ≤ M :=
by
  sorry

end smallest_valid_survey_size_l3079_307958


namespace complex_fraction_equality_l3079_307938

theorem complex_fraction_equality : (1 + I : ℂ) / (2 - I) = 1/5 + 3/5 * I := by sorry

end complex_fraction_equality_l3079_307938


namespace masons_father_age_l3079_307963

theorem masons_father_age :
  ∀ (mason_age sydney_age father_age : ℕ),
    mason_age = 20 →
    sydney_age = 3 * mason_age →
    father_age = sydney_age + 6 →
    father_age = 66 := by
  sorry

end masons_father_age_l3079_307963


namespace least_four_digit_multiple_of_seven_l3079_307999

theorem least_four_digit_multiple_of_seven : 
  (∀ n : ℕ, n < 1001 → n % 7 ≠ 0 ∨ n < 1000) ∧ 1001 % 7 = 0 := by
  sorry

end least_four_digit_multiple_of_seven_l3079_307999


namespace car_speed_theorem_l3079_307940

/-- Calculates the speed of a car in miles per hour -/
def car_speed (distance_yards : ℚ) (time_seconds : ℚ) (yards_per_mile : ℚ) : ℚ :=
  (distance_yards / yards_per_mile) * (3600 / time_seconds)

/-- Theorem stating that a car traveling 22 yards in 0.5 seconds has a speed of 90 miles per hour -/
theorem car_speed_theorem :
  car_speed 22 0.5 1760 = 90 := by
  sorry

end car_speed_theorem_l3079_307940


namespace x_minus_y_values_l3079_307911

theorem x_minus_y_values (x y : ℝ) (h : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) :
  x - y = -1 ∨ x - y = -7 := by
sorry

end x_minus_y_values_l3079_307911


namespace largest_root_is_four_l3079_307959

/-- The polynomial P(x) -/
def P (x r s : ℝ) : ℝ := x^6 - 12*x^5 + 40*x^4 - r*x^3 + s*x^2

/-- The line L(x) -/
def L (x d e : ℝ) : ℝ := d*x - e

/-- Theorem stating that the largest root of P(x) = L(x) is 4 -/
theorem largest_root_is_four 
  (r s d e : ℝ) 
  (h : ∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ < x₂ ∧ x₂ < x₃) ∧ 
    (∀ x : ℝ, P x r s = L x d e ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (∀ x : ℝ, (x - x₁)^2 * (x - x₂)^2 * (x - x₃) = P x r s - L x d e)) : 
  (∃ (x : ℝ), P x r s = L x d e ∧ ∀ y : ℝ, P y r s = L y d e → y ≤ x) ∧ 
  (∀ x : ℝ, P x r s = L x d e → x ≤ 4) :=
sorry

end largest_root_is_four_l3079_307959


namespace z_in_fourth_quadrant_l3079_307916

def z : ℂ := (3 - Complex.I) * (2 - Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l3079_307916


namespace expand_difference_of_squares_l3079_307975

theorem expand_difference_of_squares (x y : ℝ) : 
  (x - y + 1) * (x - y - 1) = x^2 - 2*x*y + y^2 - 1 := by
  sorry

end expand_difference_of_squares_l3079_307975


namespace expression_value_l3079_307944

theorem expression_value : 6 * (3/2 + 2/3) = 13 := by
  sorry

end expression_value_l3079_307944


namespace log_equality_implies_x_value_log_inequality_implies_x_range_l3079_307982

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the conditions
variable (a : ℝ)
variable (x : ℝ)
variable (h1 : a > 0)
variable (h2 : a ≠ 1)

-- Theorem 1
theorem log_equality_implies_x_value :
  log a (3*x + 1) = log a (-3*x) → x = -1/6 :=
by sorry

-- Theorem 2
theorem log_inequality_implies_x_range :
  log a (3*x + 1) > log a (-3*x) →
  ((0 < a ∧ a < 1 → -1/3 < x ∧ x < -1/6) ∧
   (a > 1 → -1/6 < x ∧ x < 0)) :=
by sorry

end log_equality_implies_x_value_log_inequality_implies_x_range_l3079_307982


namespace min_triangle_forming_number_l3079_307970

def CanFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def MinTriangleForming : ℕ → Prop
| n => ∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 1000) →
       ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ CanFormTriangle a b c

theorem min_triangle_forming_number : MinTriangleForming 16 ∧ ∀ k < 16, ¬MinTriangleForming k :=
  sorry

end min_triangle_forming_number_l3079_307970


namespace existence_of_A_l3079_307930

/-- An increasing sequence of positive integers -/
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The growth condition for the sequence -/
def GrowthCondition (a : ℕ → ℕ) (M : ℝ) : Prop :=
  ∀ n : ℕ, 0 < a (n + 1) - a n ∧ (a (n + 1) - a n : ℝ) < M * (a n : ℝ) ^ (5/8)

/-- The main theorem -/
theorem existence_of_A (a : ℕ → ℕ) (M : ℝ) 
    (h_inc : IncreasingSequence a) 
    (h_growth : GrowthCondition a M) :
    ∃ A : ℝ, ∀ k : ℕ, ∃ n : ℕ, ⌊A ^ (3^k)⌋ = a n := by
  sorry

end existence_of_A_l3079_307930
