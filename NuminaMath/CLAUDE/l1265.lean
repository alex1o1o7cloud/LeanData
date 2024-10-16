import Mathlib

namespace NUMINAMATH_CALUDE_kelly_sony_games_left_l1265_126549

/-- Given that Kelly has 132 Sony games and gives away 101 Sony games, 
    prove that she will have 31 Sony games left. -/
theorem kelly_sony_games_left : 
  ∀ (initial_sony_games given_away_sony_games : ℕ),
  initial_sony_games = 132 →
  given_away_sony_games = 101 →
  initial_sony_games - given_away_sony_games = 31 :=
by sorry

end NUMINAMATH_CALUDE_kelly_sony_games_left_l1265_126549


namespace NUMINAMATH_CALUDE_theater_tickets_l1265_126513

theorem theater_tickets (orchestra_price balcony_price : ℕ) 
  (total_tickets total_cost : ℕ) : 
  orchestra_price = 12 →
  balcony_price = 8 →
  total_tickets = 380 →
  total_cost = 3320 →
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost ∧
    balcony_tickets - orchestra_tickets = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_tickets_l1265_126513


namespace NUMINAMATH_CALUDE_ratio_equality_l1265_126543

theorem ratio_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1265_126543


namespace NUMINAMATH_CALUDE_non_trivial_solutions_l1265_126554

theorem non_trivial_solutions (a b : ℝ) : 
  (∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a^2 + b^2 = 2*a*b) ∧ 
  (∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a^2 + b^2 = 3*(a+b)) ∧ 
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ∧
  (∀ a b : ℝ, a^2 + b^2 = (a+b)^2 → a = 0 ∨ b = 0) :=
by sorry


end NUMINAMATH_CALUDE_non_trivial_solutions_l1265_126554


namespace NUMINAMATH_CALUDE_solution_set_abs_x_times_one_minus_two_x_l1265_126588

theorem solution_set_abs_x_times_one_minus_two_x (x : ℝ) :
  (|x| * (1 - 2*x) > 0) ↔ (x < 0 ∨ (x > 0 ∧ x < 1/2)) := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_times_one_minus_two_x_l1265_126588


namespace NUMINAMATH_CALUDE_ticket_distribution_count_l1265_126514

/-- Represents a valid ticket distribution for a class -/
structure TicketDistribution :=
  (min : ℕ)
  (max : ℕ)

/-- The total number of tickets to be distributed -/
def total_tickets : ℕ := 18

/-- The ticket distribution constraints for each class -/
def class_constraints : List TicketDistribution := [
  ⟨1, 5⟩,  -- Class A
  ⟨1, 6⟩,  -- Class B
  ⟨2, 7⟩,  -- Class C
  ⟨4, 10⟩  -- Class D
]

/-- 
  Counts the number of ways to distribute tickets according to the given constraints
  @param total The total number of tickets to distribute
  @param constraints The list of constraints for each class
  @return The number of valid distributions
-/
def count_distributions (total : ℕ) (constraints : List TicketDistribution) : ℕ :=
  sorry  -- Proof implementation goes here

/-- The main theorem stating that there are 140 ways to distribute the tickets -/
theorem ticket_distribution_count : count_distributions total_tickets class_constraints = 140 :=
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_ticket_distribution_count_l1265_126514


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1265_126568

theorem arithmetic_calculation : -12 * 5 - (-8 * -4) + (-15 * -6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1265_126568


namespace NUMINAMATH_CALUDE_outfit_choices_l1265_126504

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each item -/
def num_colors : ℕ := 8

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_items * num_items * num_items

/-- The number of outfits where all items are the same color -/
def mono_color_outfits : ℕ := num_colors

/-- The number of acceptable outfit choices -/
def acceptable_outfits : ℕ := total_outfits - mono_color_outfits

theorem outfit_choices : acceptable_outfits = 504 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_l1265_126504


namespace NUMINAMATH_CALUDE_saras_quarters_l1265_126538

/-- Given that Sara initially had 783 quarters and now has 1054 quarters,
    prove that the number of quarters Sara's dad gave her is 271. -/
theorem saras_quarters (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 783)
  (h2 : final_quarters = 1054) :
  final_quarters - initial_quarters = 271 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l1265_126538


namespace NUMINAMATH_CALUDE_product_digit_count_l1265_126536

def number1 : ℕ := 925743857234987123123
def number2 : ℕ := 10345678909876

theorem product_digit_count : (String.length (toString (number1 * number2))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_count_l1265_126536


namespace NUMINAMATH_CALUDE_infinitely_many_a_composite_sum_l1265_126515

theorem infinitely_many_a_composite_sum : ∃ f : ℕ → ℕ, 
  (∀ k : ℕ, f k > f (k - 1)) ∧ 
  (∀ a : ℕ, ∃ m : ℕ, a = f m) ∧
  (∀ a : ℕ, a ∈ Set.range f → ∀ n : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n^4 + a = x * y) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_a_composite_sum_l1265_126515


namespace NUMINAMATH_CALUDE_probability_of_selection_for_six_choose_two_l1265_126510

/-- The probability of choosing a specific person as a representative -/
def probability_of_selection (n : ℕ) (k : ℕ) : ℚ :=
  (n - 1).choose (k - 1) / n.choose k

/-- The problem statement -/
theorem probability_of_selection_for_six_choose_two :
  probability_of_selection 6 2 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_for_six_choose_two_l1265_126510


namespace NUMINAMATH_CALUDE_calculation_proof_l1265_126598

theorem calculation_proof : 
  (1.25 * (2/9 : ℚ)) + ((10/9 : ℚ) * (5/4 : ℚ)) - ((125/100 : ℚ) * (1/3 : ℚ)) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1265_126598


namespace NUMINAMATH_CALUDE_hash_difference_l1265_126599

/-- The # operation -/
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

/-- Theorem stating that (5#3) - (3#5) = -8 -/
theorem hash_difference : hash 5 3 - hash 3 5 = -8 := by sorry

end NUMINAMATH_CALUDE_hash_difference_l1265_126599


namespace NUMINAMATH_CALUDE_bowling_team_size_l1265_126521

/-- The number of players in a bowling team -/
def num_players : ℕ := sorry

/-- The league record average score per player per round -/
def league_record : ℕ := 287

/-- The number of rounds in a season -/
def num_rounds : ℕ := 10

/-- The team's current total score after 9 rounds -/
def current_score : ℕ := 10440

/-- The difference between the league record and the minimum average needed in the final round -/
def final_round_diff : ℕ := 27

theorem bowling_team_size :
  (num_players * league_record * num_rounds - current_score) / num_players = 
  league_record - final_round_diff ∧
  num_players = 4 := by sorry

end NUMINAMATH_CALUDE_bowling_team_size_l1265_126521


namespace NUMINAMATH_CALUDE_mushroom_drying_l1265_126509

/-- Given an initial mass of mushrooms and moisture contents before and after drying,
    calculate the mass of mushrooms after drying. -/
theorem mushroom_drying (initial_mass : ℝ) (initial_moisture : ℝ) (final_moisture : ℝ) :
  initial_mass = 100 →
  initial_moisture = 99 / 100 →
  final_moisture = 98 / 100 →
  (1 - initial_moisture) * initial_mass / (1 - final_moisture) = 50 := by
  sorry

#check mushroom_drying

end NUMINAMATH_CALUDE_mushroom_drying_l1265_126509


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l1265_126540

theorem geometric_series_r_value (a r : ℝ) (h1 : a ≠ 0) (h2 : |r| < 1) :
  (a / (1 - r) = 20) →
  (a * r / (1 - r^2) = 8) →
  r = 2/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l1265_126540


namespace NUMINAMATH_CALUDE_inequality_proof_l1265_126539

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1265_126539


namespace NUMINAMATH_CALUDE_worker_count_proof_l1265_126519

/-- The number of workers who raised money by equal contribution -/
def number_of_workers : ℕ := 1200

/-- The total contribution in rupees -/
def total_contribution : ℕ := 300000

/-- The increased total contribution if each worker contributed 50 rupees extra -/
def increased_contribution : ℕ := 360000

/-- The extra amount each worker would contribute in the increased scenario -/
def extra_contribution : ℕ := 50

theorem worker_count_proof :
  (number_of_workers * (total_contribution / number_of_workers) = total_contribution) ∧
  (number_of_workers * (total_contribution / number_of_workers + extra_contribution) = increased_contribution) :=
sorry

end NUMINAMATH_CALUDE_worker_count_proof_l1265_126519


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1265_126518

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ a + b ≤ x + y ∧ a + b = 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1265_126518


namespace NUMINAMATH_CALUDE_division_of_powers_l1265_126572

theorem division_of_powers (n : ℕ) : 19^11 / 19^8 = 6859 := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l1265_126572


namespace NUMINAMATH_CALUDE_bisecting_centers_form_line_l1265_126594

/-- Two non-overlapping circles in a plane -/
structure TwoCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  R₁ : ℝ
  R₂ : ℝ
  h_positive : R₁ > 0 ∧ R₂ > 0
  h_non_overlapping : Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) > R₁ + R₂

/-- A point that is the center of a circle bisecting both given circles -/
def BisectingCenter (tc : TwoCircles) (X : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    r^2 = (X.1 - tc.O₁.1)^2 + (X.2 - tc.O₁.2)^2 + tc.R₁^2 ∧
    r^2 = (X.1 - tc.O₂.1)^2 + (X.2 - tc.O₂.2)^2 + tc.R₂^2

/-- The locus of bisecting centers forms a straight line -/
theorem bisecting_centers_form_line (tc : TwoCircles) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
    (∀ X : ℝ × ℝ, BisectingCenter tc X ↔ a * X.1 + b * X.2 + c = 0) ∧
    (a * (tc.O₂.1 - tc.O₁.1) + b * (tc.O₂.2 - tc.O₁.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_bisecting_centers_form_line_l1265_126594


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1265_126500

def U : Set Int := Set.univ

def A : Set Int := {-1, 1, 3, 5, 7, 9}

def B : Set Int := {-1, 5, 7}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1265_126500


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l1265_126564

-- Define the square and triangle
def square_side_length : ℝ := 16
def triangle_leg_length : ℝ := 8

-- Define the theorem
theorem perimeter_of_modified_square :
  let square_perimeter := 4 * square_side_length
  let triangle_hypotenuse := Real.sqrt (2 * triangle_leg_length ^ 2)
  let new_figure_perimeter := square_perimeter - triangle_leg_length + triangle_hypotenuse
  new_figure_perimeter = 64 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_modified_square_l1265_126564


namespace NUMINAMATH_CALUDE_sin_pi_sixth_minus_2alpha_l1265_126547

theorem sin_pi_sixth_minus_2alpha (α : ℝ) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sixth_minus_2alpha_l1265_126547


namespace NUMINAMATH_CALUDE_equation_solution_l1265_126526

theorem equation_solution : ∃ x : ℝ, x = 37/10 ∧ Real.sqrt (3 * Real.sqrt (x - 3)) = (10 - x) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1265_126526


namespace NUMINAMATH_CALUDE_commute_time_difference_commute_time_difference_is_two_l1265_126525

/-- The difference in commute time between walking and taking the train -/
theorem commute_time_difference : ℝ :=
  let distance : ℝ := 1.5  -- miles
  let walking_speed : ℝ := 3  -- mph
  let train_speed : ℝ := 20  -- mph
  let additional_train_time : ℝ := 23.5  -- minutes

  let walking_time : ℝ := distance / walking_speed * 60  -- minutes
  let train_travel_time : ℝ := distance / train_speed * 60  -- minutes
  let total_train_time : ℝ := train_travel_time + additional_train_time

  walking_time - total_train_time

/-- The commute time difference is 2 minutes -/
theorem commute_time_difference_is_two : commute_time_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_commute_time_difference_is_two_l1265_126525


namespace NUMINAMATH_CALUDE_zeros_in_5000_to_50_l1265_126557

theorem zeros_in_5000_to_50 : ∃ n : ℕ, (5000 ^ 50 : ℕ) = n * (10 ^ 150) ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_5000_to_50_l1265_126557


namespace NUMINAMATH_CALUDE_order_of_values_l1265_126541

theorem order_of_values : 
  let a := Real.sin (60 * π / 180)
  let b := Real.sqrt (5 / 9)
  let c := π / 2014
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_order_of_values_l1265_126541


namespace NUMINAMATH_CALUDE_island_not_maya_l1265_126575

-- Define the possible states for an inhabitant
inductive InhabitantState
  | Knight
  | Knave

-- Define the island name
structure IslandName where
  name : String

-- Define the statements made by the inhabitants
def statement_A (state_A state_B : InhabitantState) (island : IslandName) : Prop :=
  (state_A = InhabitantState.Knave ∨ state_B = InhabitantState.Knave) ∧ island.name = "Maya"

def statement_B (state_A state_B : InhabitantState) (island : IslandName) : Prop :=
  statement_A state_A state_B island

-- Define the truthfulness of statements based on the inhabitant's state
def is_truthful (state : InhabitantState) (statement : Prop) : Prop :=
  (state = InhabitantState.Knight ∧ statement) ∨ (state = InhabitantState.Knave ∧ ¬statement)

-- Theorem statement
theorem island_not_maya (state_A state_B : InhabitantState) (island : IslandName) :
  (is_truthful state_A (statement_A state_A state_B island) ∧
   is_truthful state_B (statement_B state_A state_B island)) →
  island.name ≠ "Maya" :=
by sorry

end NUMINAMATH_CALUDE_island_not_maya_l1265_126575


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l1265_126550

theorem mark_and_carolyn_money : 
  let mark_money : ℚ := 3/4
  let carolyn_money : ℚ := 3/10
  mark_money + carolyn_money = 21/20 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l1265_126550


namespace NUMINAMATH_CALUDE_inverse_variation_l1265_126535

/-- Given quantities a and b that vary inversely, if b = 0.5 when a = 800, 
    then b = 0.25 when a = 1600 -/
theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) 
  (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l1265_126535


namespace NUMINAMATH_CALUDE_tank_fill_time_with_leak_l1265_126527

def pump_rate : ℚ := 1 / 6
def leak_rate : ℚ := 1 / 12

theorem tank_fill_time_with_leak :
  let net_fill_rate := pump_rate - leak_rate
  (1 : ℚ) / net_fill_rate = 12 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_with_leak_l1265_126527


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l1265_126524

/-- Given that 2, b, a form a geometric sequence in order, prove that the equation ax^2 + bx + 1/3 = 0 has exactly 2 real roots -/
theorem geometric_sequence_quadratic_roots
  (b a : ℝ)
  (h_geometric : ∃ (q : ℝ), b = 2 * q ∧ a = 2 * q^2) :
  (∃! (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + 1/3 = 0 ∧ a * y^2 + b * y + 1/3 = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l1265_126524


namespace NUMINAMATH_CALUDE_farmhouse_blocks_l1265_126582

def total_blocks : ℕ := 344
def building_blocks : ℕ := 80
def fenced_area_blocks : ℕ := 57
def leftover_blocks : ℕ := 84

theorem farmhouse_blocks :
  total_blocks - building_blocks - fenced_area_blocks - leftover_blocks = 123 := by
  sorry

end NUMINAMATH_CALUDE_farmhouse_blocks_l1265_126582


namespace NUMINAMATH_CALUDE_average_equation_l1265_126546

theorem average_equation (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 4) + (4*x + 6) + (5*x + 3)) = 3*x + 5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l1265_126546


namespace NUMINAMATH_CALUDE_real_part_of_z_l1265_126503

theorem real_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 3 - 2*I) : 
  z.re = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1265_126503


namespace NUMINAMATH_CALUDE_circle_condition_l1265_126579

/-- The equation of a circle in terms of parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0

/-- Theorem stating the condition for m to represent a circle -/
theorem circle_condition (m : ℝ) : 
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1265_126579


namespace NUMINAMATH_CALUDE_x_value_l1265_126585

theorem x_value (x y z : ℤ) 
  (eq1 : x + y + z = 14)
  (eq2 : x - y - z = 60)
  (eq3 : x + z = 2*y) : 
  x = 37 := by
sorry

end NUMINAMATH_CALUDE_x_value_l1265_126585


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1265_126563

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : (Real.tan α) ^ 2 + 2 * (Real.tan α) - 3 = 0) : 
  Real.sin α = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1265_126563


namespace NUMINAMATH_CALUDE_circumradius_leg_ratio_not_always_equal_l1265_126555

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ

/-- The ratio of circumradii is not always equal to the ratio of leg lengths for two isosceles triangles with different leg lengths -/
theorem circumradius_leg_ratio_not_always_equal 
  (t1 t2 : IsoscelesTriangle) 
  (h : t1.leg ≠ t2.leg) : 
  ¬ ∀ (t1 t2 : IsoscelesTriangle), t1.circumradius / t2.circumradius = t1.leg / t2.leg :=
by sorry

end NUMINAMATH_CALUDE_circumradius_leg_ratio_not_always_equal_l1265_126555


namespace NUMINAMATH_CALUDE_dot_product_range_l1265_126511

-- Define the unit circle
def unit_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define vector AO
def vector_AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)

-- Define vector AP
def vector_AP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_range :
  ∀ P : ℝ × ℝ, unit_circle P →
    2 ≤ dot_product vector_AO (vector_AP P) ∧
    dot_product vector_AO (vector_AP P) ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l1265_126511


namespace NUMINAMATH_CALUDE_trig_identity_l1265_126595

theorem trig_identity (α : Real) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1265_126595


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_l1265_126559

/-- A direct proportion function passing through (2, -1) -/
def f (x : ℝ) : ℝ := sorry

/-- The point (2, -1) lies on the graph of f -/
axiom point_on_graph : f 2 = -1

/-- f is a direct proportion function -/
axiom direct_proportion (x : ℝ) : ∃ k : ℝ, f x = k * x

theorem direct_proportion_through_point :
  ∀ x : ℝ, f x = -1/2 * x := by sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_l1265_126559


namespace NUMINAMATH_CALUDE_complex_simplification_l1265_126537

/-- Proof of complex number simplification -/
theorem complex_simplification :
  let i : ℂ := Complex.I
  (3 + 5*i) / (-2 + 7*i) = 29/53 - (31/53)*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1265_126537


namespace NUMINAMATH_CALUDE_new_person_weight_l1265_126552

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 35 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1265_126552


namespace NUMINAMATH_CALUDE_integral_arctan_fraction_l1265_126530

open Real

theorem integral_arctan_fraction (x : ℝ) :
  deriv (fun x => (1/2) * (4 * (arctan x)^2 - log (1 + x^2))) x
  = (4 * arctan x - x) / (1 + x^2) :=
by sorry

end NUMINAMATH_CALUDE_integral_arctan_fraction_l1265_126530


namespace NUMINAMATH_CALUDE_cake_remaining_l1265_126517

theorem cake_remaining (alex_portion jordan_portion remaining_portion : ℚ) : 
  alex_portion = 40 / 100 →
  jordan_portion = (1 - alex_portion) / 2 →
  remaining_portion = 1 - alex_portion - jordan_portion →
  remaining_portion = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_remaining_l1265_126517


namespace NUMINAMATH_CALUDE_sum_of_integers_l1265_126520

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1265_126520


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l1265_126529

/-- The marshmallow challenge theorem -/
theorem marshmallow_challenge (haley michael brandon : ℕ) : 
  haley = 8 →
  michael = 3 * haley →
  brandon = michael / 2 →
  haley + michael + brandon = 44 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l1265_126529


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1265_126565

theorem binomial_coefficient_two (n : ℕ) (h : n > 1) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1265_126565


namespace NUMINAMATH_CALUDE_angle_expression_value_l1265_126589

theorem angle_expression_value (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l1265_126589


namespace NUMINAMATH_CALUDE_cost_per_pill_is_five_l1265_126591

/-- Represents the annual costs and medication details for Tom --/
structure AnnualMedication where
  pillsPerDay : ℕ
  doctorVisitsPerYear : ℕ
  doctorVisitCost : ℕ
  insuranceCoveragePercent : ℚ
  totalAnnualCost : ℕ

/-- Calculates the cost per pill before insurance coverage --/
def costPerPillBeforeInsurance (am : AnnualMedication) : ℚ :=
  let totalPillsPerYear := am.pillsPerDay * 365
  let annualDoctorVisitsCost := am.doctorVisitsPerYear * am.doctorVisitCost
  let annualMedicationCost := am.totalAnnualCost - annualDoctorVisitsCost
  let totalMedicationCostBeforeInsurance := annualMedicationCost / (1 - am.insuranceCoveragePercent)
  totalMedicationCostBeforeInsurance / totalPillsPerYear

/-- Theorem stating that the cost per pill before insurance is $5 --/
theorem cost_per_pill_is_five (am : AnnualMedication) 
    (h1 : am.pillsPerDay = 2)
    (h2 : am.doctorVisitsPerYear = 2)
    (h3 : am.doctorVisitCost = 400)
    (h4 : am.insuranceCoveragePercent = 4/5)
    (h5 : am.totalAnnualCost = 1530) : 
  costPerPillBeforeInsurance am = 5 := by
  sorry

#eval costPerPillBeforeInsurance {
  pillsPerDay := 2,
  doctorVisitsPerYear := 2,
  doctorVisitCost := 400,
  insuranceCoveragePercent := 4/5,
  totalAnnualCost := 1530
}

end NUMINAMATH_CALUDE_cost_per_pill_is_five_l1265_126591


namespace NUMINAMATH_CALUDE_expected_black_pairs_in_circular_arrangement_l1265_126522

/-- The number of cards in the modified deck -/
def total_cards : ℕ := 60

/-- The number of black cards in the deck -/
def black_cards : ℕ := 30

/-- The number of red cards in the deck -/
def red_cards : ℕ := 30

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expected_black_pairs : ℚ := 870 / 59

theorem expected_black_pairs_in_circular_arrangement :
  let total := total_cards
  let black := black_cards
  let red := red_cards
  total = black + red →
  expected_black_pairs = (black * (black - 1) : ℚ) / (total - 1) := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_in_circular_arrangement_l1265_126522


namespace NUMINAMATH_CALUDE_inequality_condition_l1265_126567

theorem inequality_condition (x y : ℝ) : 
  (((x^3 + y^3) / 2)^(1/3) ≥ ((x^2 + y^2) / 2)^(1/2)) ↔ (x + y ≥ 0 ∨ x + y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1265_126567


namespace NUMINAMATH_CALUDE_oranges_harvested_proof_l1265_126507

/-- The number of oranges harvested per day that are not discarded -/
def oranges_kept (sacks_harvested : ℕ) (sacks_discarded : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  (sacks_harvested - sacks_discarded) * oranges_per_sack

/-- Proof that the number of oranges harvested per day that are not discarded is 600 -/
theorem oranges_harvested_proof :
  oranges_kept 76 64 50 = 600 := by
  sorry

end NUMINAMATH_CALUDE_oranges_harvested_proof_l1265_126507


namespace NUMINAMATH_CALUDE_fred_seashells_l1265_126561

def seashells_problem (initial_seashells given_away_seashells : ℕ) : Prop :=
  initial_seashells - given_away_seashells = 22

theorem fred_seashells : seashells_problem 47 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_seashells_l1265_126561


namespace NUMINAMATH_CALUDE_tan_negative_405_l1265_126570

-- Define the tangent function
noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ

-- Define the property of tangent periodicity
axiom tan_periodic (θ : ℝ) (n : ℤ) : tan θ = tan (θ + n * 360)

-- Define the value of tan(45°)
axiom tan_45 : tan 45 = 1

-- Theorem to prove
theorem tan_negative_405 : tan (-405) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_l1265_126570


namespace NUMINAMATH_CALUDE_valid_numbers_l1265_126569

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) / 143 = 136

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {9949, 9859, 9769, 9679, 9589, 9499} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1265_126569


namespace NUMINAMATH_CALUDE_units_c_is_twelve_l1265_126573

/-- Represents the sales data for a salesperson --/
structure SalesData where
  commissionRateA : ℝ
  commissionRateB : ℝ
  commissionRateC : ℝ
  unitsA : ℕ
  unitsB : ℕ
  revenueA : ℝ
  revenueB : ℝ
  revenueC : ℝ
  avgCommissionIncrease : ℝ
  newAvgCommission : ℝ

/-- Calculates the number of units of Product C sold --/
def calculateUnitsC (data : SalesData) : ℕ :=
  sorry

/-- Theorem stating that given the sales data, the number of units of Product C sold is 12 --/
theorem units_c_is_twelve (data : SalesData) 
  (h1 : data.commissionRateA = 0.05)
  (h2 : data.commissionRateB = 0.07)
  (h3 : data.commissionRateC = 0.10)
  (h4 : data.unitsA = 5)
  (h5 : data.unitsB = 3)
  (h6 : data.revenueA = 1500)
  (h7 : data.revenueB = 2000)
  (h8 : data.revenueC = 3500)
  (h9 : data.avgCommissionIncrease = 150)
  (h10 : data.newAvgCommission = 250) :
  calculateUnitsC data = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_units_c_is_twelve_l1265_126573


namespace NUMINAMATH_CALUDE_cat_stolen_pieces_l1265_126548

/-- Proves the number of pieces the cat stole given the conditions of the problem -/
theorem cat_stolen_pieces (total : ℕ) (boyfriendPieces : ℕ) : 
  total = 60 ∧ 
  boyfriendPieces = 9 ∧ 
  boyfriendPieces = (total - total / 2) / 3 →
  total - (total / 2) - ((total - total / 2) / 3) - boyfriendPieces = 3 :=
by sorry

end NUMINAMATH_CALUDE_cat_stolen_pieces_l1265_126548


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1265_126501

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≥ 
  Real.sqrt (3 / 2) * Real.sqrt (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1265_126501


namespace NUMINAMATH_CALUDE_max_m_value_l1265_126516

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_m_value :
  (∃ (m : ℝ), ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = m) →
  (∀ (m : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = m) → m ≤ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_max_m_value_l1265_126516


namespace NUMINAMATH_CALUDE_edward_savings_l1265_126571

/-- Represents the amount of money Edward had saved before mowing lawns -/
def money_saved (earnings_per_lawn : ℕ) (lawns_mowed : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (earnings_per_lawn * lawns_mowed)

/-- Theorem stating that Edward's savings before mowing can be calculated -/
theorem edward_savings :
  let earnings_per_lawn : ℕ := 8
  let lawns_mowed : ℕ := 5
  let total_money : ℕ := 47
  money_saved earnings_per_lawn lawns_mowed total_money = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_savings_l1265_126571


namespace NUMINAMATH_CALUDE_identity_proof_l1265_126580

theorem identity_proof (a b x y θ φ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h1 : (a - b) * Real.sin (θ / 2) * Real.cos (φ / 2) + 
        (a + b) * Real.cos (θ / 2) * Real.sin (φ / 2) = 0)
  (h2 : x / a * Real.cos θ + y / b * Real.sin θ = 1)
  (h3 : x / a * Real.cos φ + y / b * Real.sin φ = 1) :
  x^2 / a^2 + (b^2 - a^2) / b^4 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_identity_proof_l1265_126580


namespace NUMINAMATH_CALUDE_unique_I_value_l1265_126542

def addition_problem (E I G T W O : Nat) : Prop :=
  E ≠ I ∧ E ≠ G ∧ E ≠ T ∧ E ≠ W ∧ E ≠ O ∧
  I ≠ G ∧ I ≠ T ∧ I ≠ W ∧ I ≠ O ∧
  G ≠ T ∧ G ≠ W ∧ G ≠ O ∧
  T ≠ W ∧ T ≠ O ∧
  W ≠ O ∧
  E < 10 ∧ I < 10 ∧ G < 10 ∧ T < 10 ∧ W < 10 ∧ O < 10 ∧
  E = 4 ∧
  G % 2 = 1 ∧
  100 * T + 10 * W + O = 100 * E + 10 * I + G + 100 * E + 10 * I + G

theorem unique_I_value :
  ∀ E I G T W O : Nat,
    addition_problem E I G T W O →
    I = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_I_value_l1265_126542


namespace NUMINAMATH_CALUDE_percentage_of_indian_men_l1265_126562

theorem percentage_of_indian_men (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_women : ℚ) (percent_indian_children : ℚ) (percent_non_indian : ℚ)
  (h1 : total_men = 700)
  (h2 : total_women = 500)
  (h3 : total_children = 800)
  (h4 : percent_indian_women = 40 / 100)
  (h5 : percent_indian_children = 10 / 100)
  (h6 : percent_non_indian = 79 / 100) :
  (↑(total_men + total_women + total_children) * (1 - percent_non_indian) -
   ↑total_women * percent_indian_women -
   ↑total_children * percent_indian_children) / ↑total_men = 20 / 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_indian_men_l1265_126562


namespace NUMINAMATH_CALUDE_cos_equality_for_n_l1265_126576

theorem cos_equality_for_n (n : ℤ) : ∃ n, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (259 * π / 180) ∧ n = 101 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_for_n_l1265_126576


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1265_126558

/-- A positive geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ) (q : ℝ) (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  GeometricSequence a q →
  (a m * a n).sqrt = 4 * a 1 →
  a 7 = a 6 + 2 * a 5 →
  (1 : ℝ) / m + 5 / n ≥ 7 / 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1265_126558


namespace NUMINAMATH_CALUDE_xy_value_l1265_126512

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1265_126512


namespace NUMINAMATH_CALUDE_f_3_minus_f_4_l1265_126545

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_3_minus_f_4 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 5)
  (h_f_1 : f 1 = 1)
  (h_f_2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
sorry

end NUMINAMATH_CALUDE_f_3_minus_f_4_l1265_126545


namespace NUMINAMATH_CALUDE_power_equality_implies_y_equals_four_l1265_126596

theorem power_equality_implies_y_equals_four :
  ∀ y : ℝ, (4 : ℝ)^12 = 64^y → y = 4 := by
sorry

end NUMINAMATH_CALUDE_power_equality_implies_y_equals_four_l1265_126596


namespace NUMINAMATH_CALUDE_prob_two_unmarked_correct_l1265_126586

/-- The probability of selecting two unmarked items from a set of 10 items where 3 are marked -/
def prob_two_unmarked (total : Nat) (marked : Nat) (select : Nat) : Rat :=
  if total = 10 ∧ marked = 3 ∧ select = 2 then
    7 / 15
  else
    0

theorem prob_two_unmarked_correct :
  prob_two_unmarked 10 3 2 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_unmarked_correct_l1265_126586


namespace NUMINAMATH_CALUDE_subtract_from_21_to_get_8_l1265_126581

theorem subtract_from_21_to_get_8 : ∃ x : ℝ, 21 - x = 8 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_21_to_get_8_l1265_126581


namespace NUMINAMATH_CALUDE_painted_cubes_ratio_l1265_126551

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with exactly two painted faces in a rectangular prism -/
def count_two_faces (prism : RectangularPrism) : ℕ :=
  4 * ((prism.length - 2) + (prism.width - 2) + (prism.height - 2))

/-- Counts the number of cubes with exactly three painted faces in a rectangular prism -/
def count_three_faces (prism : RectangularPrism) : ℕ := 8

/-- The main theorem statement -/
theorem painted_cubes_ratio (prism : RectangularPrism)
    (h_length : prism.length = 4)
    (h_width : prism.width = 5)
    (h_height : prism.height = 6) :
    (count_two_faces prism) / (count_three_faces prism) = 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_painted_cubes_ratio_l1265_126551


namespace NUMINAMATH_CALUDE_total_pennies_thrown_l1265_126577

/-- The number of pennies thrown by each person -/
structure PennyThrowers where
  rachelle : ℕ
  gretchen : ℕ
  rocky : ℕ
  max : ℕ
  taylor : ℕ

/-- The conditions of the penny-throwing problem -/
def penny_throwing_conditions (pt : PennyThrowers) : Prop :=
  pt.rachelle = 720 ∧
  pt.gretchen = pt.rachelle / 2 ∧
  pt.rocky = pt.gretchen / 3 ∧
  pt.max = pt.rocky * 4 ∧
  pt.taylor = pt.max / 5

/-- The theorem stating that the total number of pennies thrown is 1776 -/
theorem total_pennies_thrown (pt : PennyThrowers) 
  (h : penny_throwing_conditions pt) : 
  pt.rachelle + pt.gretchen + pt.rocky + pt.max + pt.taylor = 1776 := by
  sorry


end NUMINAMATH_CALUDE_total_pennies_thrown_l1265_126577


namespace NUMINAMATH_CALUDE_grocer_banana_purchase_l1265_126534

/-- Calculates the number of pounds of bananas purchased by a grocer given the purchase price, selling price, and total profit. -/
theorem grocer_banana_purchase
  (purchase_price : ℚ)
  (purchase_quantity : ℚ)
  (selling_price : ℚ)
  (selling_quantity : ℚ)
  (total_profit : ℚ)
  (h1 : purchase_price / purchase_quantity = 0.50 / 3)
  (h2 : selling_price / selling_quantity = 1.00 / 4)
  (h3 : total_profit = 9.00) :
  ∃ (pounds : ℚ), pounds = 108 ∧ 
    pounds * (selling_price / selling_quantity - purchase_price / purchase_quantity) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_grocer_banana_purchase_l1265_126534


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_price_l1265_126531

/-- Represents the price Ramesh paid for a refrigerator given certain conditions --/
def ramesh_paid_price (P : ℝ) : Prop :=
  let discount_rate : ℝ := 0.20
  let transport_cost : ℝ := 125
  let installation_cost : ℝ := 250
  let profit_rate : ℝ := 0.10
  let selling_price : ℝ := 20350
  (1 + profit_rate) * P = selling_price ∧
  (1 - discount_rate) * P + transport_cost + installation_cost = 15175

theorem ramesh_refrigerator_price :
  ∃ P : ℝ, ramesh_paid_price P :=
sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_price_l1265_126531


namespace NUMINAMATH_CALUDE_simplify_expression_l1265_126544

theorem simplify_expression (x : ℝ) : 2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 8) = 8*x^3 - 4*x^2 + 6*x - 32 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1265_126544


namespace NUMINAMATH_CALUDE_binomial_18_choose_7_l1265_126560

theorem binomial_18_choose_7 : Nat.choose 18 7 = 31824 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_7_l1265_126560


namespace NUMINAMATH_CALUDE_percentage_relation_l1265_126583

theorem percentage_relation (x a b : ℝ) (ha : a = 0.07 * x) (hb : b = 0.14 * x) :
  a / b = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l1265_126583


namespace NUMINAMATH_CALUDE_second_book_has_32_pictures_l1265_126532

/-- The number of pictures in the second coloring book -/
def second_book_pictures (first_book_pictures colored_pictures remaining_pictures : ℕ) : ℕ :=
  (colored_pictures + remaining_pictures) - first_book_pictures

/-- Theorem stating that the second coloring book has 32 pictures -/
theorem second_book_has_32_pictures :
  second_book_pictures 23 44 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_second_book_has_32_pictures_l1265_126532


namespace NUMINAMATH_CALUDE_inequality_proof_l1265_126590

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(a+c)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1265_126590


namespace NUMINAMATH_CALUDE_equality_of_ratios_implies_k_eighteen_l1265_126578

theorem equality_of_ratios_implies_k_eighteen 
  (x y z k : ℝ) 
  (h : (7 : ℝ) / (x + y) = k / (x + z) ∧ k / (x + z) = (11 : ℝ) / (z - y)) : 
  k = 18 := by
sorry

end NUMINAMATH_CALUDE_equality_of_ratios_implies_k_eighteen_l1265_126578


namespace NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l1265_126553

theorem half_plus_five_equals_thirteen (n : ℝ) : (1/2 : ℝ) * n + 5 = 13 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l1265_126553


namespace NUMINAMATH_CALUDE_blue_beads_count_l1265_126587

theorem blue_beads_count (total : ℕ) (blue_neighbors : ℕ) (green_neighbors : ℕ) :
  total = 30 →
  blue_neighbors = 26 →
  green_neighbors = 20 →
  ∃ blue_count : ℕ,
    blue_count = 18 ∧
    blue_count ≤ total ∧
    blue_count * 2 ≥ blue_neighbors ∧
    (total - blue_count) * 2 ≥ green_neighbors :=
by
  sorry


end NUMINAMATH_CALUDE_blue_beads_count_l1265_126587


namespace NUMINAMATH_CALUDE_max_gold_coins_max_gold_coins_proof_l1265_126523

/-- The largest number of gold coins that can be distributed among 15 friends
    with 4 coins left over and a total less than 150. -/
theorem max_gold_coins : ℕ :=
  let num_friends : ℕ := 15
  let extra_coins : ℕ := 4
  let max_total : ℕ := 149  -- less than 150
  
  have h1 : ∃ (k : ℕ), num_friends * k + extra_coins ≤ max_total :=
    sorry
  
  have h2 : ∀ (n : ℕ), num_friends * n + extra_coins > max_total → n > 9 :=
    sorry
  
  139

theorem max_gold_coins_proof (n : ℕ) :
  n ≤ max_gold_coins ∧
  (∃ (k : ℕ), n = 15 * k + 4) ∧
  n < 150 :=
by sorry

end NUMINAMATH_CALUDE_max_gold_coins_max_gold_coins_proof_l1265_126523


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l1265_126592

/-- Given a curve y = e^(ax), prove that if its tangent line at (0,1) is perpendicular to the line x + 2y + 1 = 0, then a = 2. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  (∀ x, deriv (fun x => Real.exp (a * x)) x = a * Real.exp (a * x)) →
  (fun x => Real.exp (a * x)) 0 = 1 →
  (deriv (fun x => Real.exp (a * x))) 0 = (-1 / (2 : ℝ))⁻¹ →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l1265_126592


namespace NUMINAMATH_CALUDE_lasagna_cost_l1265_126505

def cheese_quantity : Real := 1.5
def meat_quantity : Real := 0.550
def pasta_quantity : Real := 0.280
def tomatoes_quantity : Real := 2.2

def cheese_price : Real := 6.30
def meat_price : Real := 8.55
def pasta_price : Real := 2.40
def tomatoes_price : Real := 1.79

def cheese_tax : Real := 0.07
def meat_tax : Real := 0.06
def pasta_tax : Real := 0.08
def tomatoes_tax : Real := 0.05

def total_cost (cq mq pq tq : Real) (cp mp pp tp : Real) (ct mt pt tt : Real) : Real :=
  (cq * cp * (1 + ct)) + (mq * mp * (1 + mt)) + (pq * pp * (1 + pt)) + (tq * tp * (1 + tt))

theorem lasagna_cost :
  total_cost cheese_quantity meat_quantity pasta_quantity tomatoes_quantity
              cheese_price meat_price pasta_price tomatoes_price
              cheese_tax meat_tax pasta_tax tomatoes_tax = 19.9568 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_cost_l1265_126505


namespace NUMINAMATH_CALUDE_isaac_sleep_time_l1265_126508

-- Define a simple representation of time
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

def Time.isAM (t : Time) : Bool :=
  t.hour < 12

def Time.toPM (t : Time) : Time :=
  if t.isAM then { hour := t.hour + 12, minute := t.minute }
  else t

def Time.fromPM (t : Time) : Time :=
  if t.isAM then t
  else { hour := t.hour - 12, minute := t.minute }

def subtractHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute
  let newTotalMinutes := totalMinutes - h * 60
  let newHour := newTotalMinutes / 60
  let newMinute := newTotalMinutes % 60
  { hour := newHour, minute := newMinute }

theorem isaac_sleep_time (wakeUpTime sleepTime : Time) (sleepDuration : Nat) :
  wakeUpTime = { hour := 7, minute := 0 } →
  sleepDuration = 8 →
  sleepTime = (subtractHours wakeUpTime sleepDuration).toPM →
  sleepTime = { hour := 23, minute := 0 } :=
by sorry

end NUMINAMATH_CALUDE_isaac_sleep_time_l1265_126508


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1265_126574

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 110 → 
  train_speed_kmh = 72 → 
  bridge_length = 140 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 12.5 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1265_126574


namespace NUMINAMATH_CALUDE_line_difference_l1265_126566

theorem line_difference (line_length : ℝ) (h : line_length = 80) :
  (0.75 - 0.4) * line_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_line_difference_l1265_126566


namespace NUMINAMATH_CALUDE_divisible_by_512_l1265_126593

theorem divisible_by_512 (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, n^12 - n^8 - n^4 + 1 = 512 * k := by sorry

end NUMINAMATH_CALUDE_divisible_by_512_l1265_126593


namespace NUMINAMATH_CALUDE_sample_size_accuracy_l1265_126502

theorem sample_size_accuracy (population : Type) (sample : Set population) (estimate : Set population → ℝ) (accuracy : Set population → ℝ) :
  ∀ s₁ s₂ : Set population, s₁ ⊆ s₂ → accuracy s₁ ≤ accuracy s₂ := by
  sorry

end NUMINAMATH_CALUDE_sample_size_accuracy_l1265_126502


namespace NUMINAMATH_CALUDE_combined_total_value_l1265_126556

/-- Represents the coin counts for a person -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  halfDollars : ℕ
  dollarCoins : ℕ

/-- Calculates the total value of coins for a person -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies * 1 +
  coins.nickels * 5 +
  coins.dimes * 10 +
  coins.quarters * 25 +
  coins.halfDollars * 50 +
  coins.dollarCoins * 100

/-- The coin counts for Kate -/
def kate : CoinCounts := {
  pennies := 223
  nickels := 156
  dimes := 87
  quarters := 25
  halfDollars := 7
  dollarCoins := 4
}

/-- The coin counts for John -/
def john : CoinCounts := {
  pennies := 388
  nickels := 94
  dimes := 105
  quarters := 45
  halfDollars := 15
  dollarCoins := 6
}

/-- The coin counts for Marie -/
def marie : CoinCounts := {
  pennies := 517
  nickels := 64
  dimes := 78
  quarters := 63
  halfDollars := 12
  dollarCoins := 9
}

/-- The coin counts for George -/
def george : CoinCounts := {
  pennies := 289
  nickels := 72
  dimes := 132
  quarters := 50
  halfDollars := 4
  dollarCoins := 3
}

/-- Theorem stating that the combined total value of all coins is 16042 cents -/
theorem combined_total_value :
  totalValue kate + totalValue john + totalValue marie + totalValue george = 16042 := by
  sorry

end NUMINAMATH_CALUDE_combined_total_value_l1265_126556


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l1265_126584

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ)
  (other_communities : ℕ) (hindu_percentage_condition : hindu_percentage = 28 / 100)
  (sikh_percentage_condition : sikh_percentage = 10 / 100)
  (total_boys_condition : total_boys = 850)
  (other_communities_condition : other_communities = 187) :
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_communities)) /
  total_boys * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l1265_126584


namespace NUMINAMATH_CALUDE_junior_score_l1265_126506

theorem junior_score (total : ℕ) (junior_score : ℝ) : 
  total > 0 →
  let junior_count : ℝ := 0.2 * total
  let senior_count : ℝ := 0.8 * total
  let overall_avg : ℝ := 86
  let senior_avg : ℝ := 85
  (junior_count * junior_score + senior_count * senior_avg) / total = overall_avg →
  junior_score = 90 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l1265_126506


namespace NUMINAMATH_CALUDE_system_solution_l1265_126533

theorem system_solution : ∃ (x y : ℝ), 
  (x = 4 + 2 * Real.sqrt 3 ∧ y = 12 + 6 * Real.sqrt 3) ∧
  (1 - 12 / (3 * x + y) = 2 / Real.sqrt x) ∧
  (1 + 12 / (3 * x + y) = 6 / Real.sqrt y) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1265_126533


namespace NUMINAMATH_CALUDE_second_train_length_correct_l1265_126528

/-- Calculates the length of a train given the length of another train, their speeds, and the time they take to cross each other when moving in opposite directions. -/
def calculate_train_length (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := speed_train1 + speed_train2
  let total_distance := relative_speed * time_to_cross
  total_distance - length_train1

/-- Theorem stating that the calculated length of the second train is correct given the problem conditions. -/
theorem second_train_length_correct (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ)
  (h1 : length_train1 = 110)
  (h2 : speed_train1 = 60 * 1000 / 3600)
  (h3 : speed_train2 = 40 * 1000 / 3600)
  (h4 : time_to_cross = 9.719222462203025) :
  let length_train2 := calculate_train_length length_train1 speed_train1 speed_train2 time_to_cross
  ∃ ε > 0, |length_train2 - 159.98| < ε :=
sorry

end NUMINAMATH_CALUDE_second_train_length_correct_l1265_126528


namespace NUMINAMATH_CALUDE_length_of_bd_l1265_126597

-- Define the equilateral triangle
def EquilateralTriangle (side_length : ℝ) : Prop :=
  side_length > 0

-- Define points A and C on the sides of the triangle
def PointA (a1 a2 : ℝ) (side_length : ℝ) : Prop :=
  a1 > 0 ∧ a2 > 0 ∧ a1 + a2 = side_length

def PointC (c1 c2 : ℝ) (side_length : ℝ) : Prop :=
  c1 > 0 ∧ c2 > 0 ∧ c1 + c2 = side_length

-- Define the line segment AB and BD
def LineSegments (ab bd : ℝ) : Prop :=
  ab > 0 ∧ bd > 0

-- Theorem statement
theorem length_of_bd
  (side_length : ℝ)
  (a1 a2 c1 c2 ab : ℝ)
  (h1 : EquilateralTriangle side_length)
  (h2 : PointA a1 a2 side_length)
  (h3 : PointC c1 c2 side_length)
  (h4 : LineSegments ab bd)
  (h5 : side_length = 26)
  (h6 : a1 = 3 ∧ a2 = 22)
  (h7 : c1 = 3 ∧ c2 = 23)
  (h8 : ab = 6)
  : bd = 3 := by
  sorry

end NUMINAMATH_CALUDE_length_of_bd_l1265_126597
