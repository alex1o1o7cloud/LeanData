import Mathlib

namespace exponential_function_through_point_l744_74451

theorem exponential_function_through_point (a : ℝ) : 
  (∀ x : ℝ, (fun x => a^x) x = a^x) → 
  a^2 = 4 → 
  a > 0 → 
  a ≠ 1 → 
  a = 2 := by
sorry

end exponential_function_through_point_l744_74451


namespace circle_equation_from_diameter_l744_74459

/-- Given two points as the endpoints of a circle's diameter, prove the equation of the circle -/
theorem circle_equation_from_diameter (p1 p2 : ℝ × ℝ) :
  p1 = (-1, 3) →
  p2 = (5, -5) →
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (x^2 + y^2 + a*x + b*y + c = 0) ↔
    ((x - ((p1.1 + p2.1) / 2))^2 + (y - ((p1.2 + p2.2) / 2))^2 = ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) / 4) :=
by sorry

end circle_equation_from_diameter_l744_74459


namespace percentage_of_120_to_80_l744_74418

theorem percentage_of_120_to_80 : ∃ (p : ℝ), p = (120 : ℝ) / 80 * 100 ∧ p = 150 := by
  sorry

end percentage_of_120_to_80_l744_74418


namespace solve_linear_equation_l744_74480

theorem solve_linear_equation :
  ∃ x : ℝ, 45 - 3 * x = 12 ∧ x = 11 := by
  sorry

end solve_linear_equation_l744_74480


namespace rotated_semicircle_area_l744_74450

/-- The area of a shaded figure formed by rotating a semicircle -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let α : ℝ := 20 * π / 180  -- Convert 20° to radians
  let semicircle_area : ℝ := π * R^2 / 2
  let sector_area : ℝ := 2 * R^2 * α / 2
  sector_area = 2 * π * R^2 / 9 :=
by sorry

end rotated_semicircle_area_l744_74450


namespace salary_increase_percentage_l744_74484

/-- Proves that given an initial monthly salary of $6000 and total earnings of $259200 after 3 years,
    with a salary increase occurring after 1 year, the percentage increase in salary is 30%. -/
theorem salary_increase_percentage 
  (initial_salary : ℝ) 
  (total_earnings : ℝ) 
  (increase_percentage : ℝ) :
  initial_salary = 6000 →
  total_earnings = 259200 →
  total_earnings = 12 * initial_salary + 24 * (initial_salary + initial_salary * increase_percentage / 100) →
  increase_percentage = 30 := by
  sorry

#check salary_increase_percentage

end salary_increase_percentage_l744_74484


namespace quadratic_equivalence_l744_74433

/-- The quadratic function in general form -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The quadratic function in vertex form -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f and g are equivalent -/
theorem quadratic_equivalence : ∀ x : ℝ, f x = g x := by sorry

end quadratic_equivalence_l744_74433


namespace horner_method_v2_l744_74485

def f (x : ℝ) : ℝ := 2*x^5 - x^4 + 2*x^2 + 5*x + 3

def horner_v2 (x v0 v1 : ℝ) : ℝ := v1 * x

theorem horner_method_v2 (x v0 v1 : ℝ) (hx : x = 3) (hv0 : v0 = 2) (hv1 : v1 = 5) :
  horner_v2 x v0 v1 = 15 := by
  sorry

end horner_method_v2_l744_74485


namespace intersection_product_l744_74491

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 10*y + 21 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 52 = 0

/-- Intersection point of the two circles -/
def intersection_point (x y : ℝ) : Prop :=
  circle1 x y ∧ circle2 x y

/-- The theorem stating that the product of all coordinates of intersection points is 189 -/
theorem intersection_product : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁ * y₁ * x₂ * y₂ = 189 :=
sorry

end intersection_product_l744_74491


namespace harolds_marbles_distribution_l744_74401

theorem harolds_marbles_distribution (total_marbles : ℕ) (kept_marbles : ℕ) 
  (best_friends : ℕ) (marbles_per_best_friend : ℕ) (cousins : ℕ) (marbles_per_cousin : ℕ) 
  (school_friends : ℕ) :
  total_marbles = 5000 →
  kept_marbles = 250 →
  best_friends = 3 →
  marbles_per_best_friend = 100 →
  cousins = 5 →
  marbles_per_cousin = 75 →
  school_friends = 10 →
  (total_marbles - (kept_marbles + best_friends * marbles_per_best_friend + 
    cousins * marbles_per_cousin)) / school_friends = 407 := by
  sorry

#check harolds_marbles_distribution

end harolds_marbles_distribution_l744_74401


namespace sector_arc_length_l744_74456

/-- Given a circular sector with area 2 cm² and central angle 4 radians,
    the length of the arc of the sector is 6 cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) :
  area = 2 →
  angle = 4 →
  arc_length = 6 :=
by sorry

end sector_arc_length_l744_74456


namespace division_preserves_inequality_l744_74489

theorem division_preserves_inequality (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end division_preserves_inequality_l744_74489


namespace sum_of_x_values_l744_74413

theorem sum_of_x_values (x : ℝ) : 
  (|x - 25| = 50) → (∃ y : ℝ, |y - 25| = 50 ∧ x + y = 50) :=
by sorry

end sum_of_x_values_l744_74413


namespace sum_of_integers_l744_74437

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 5)
  (eq4 : d - a + b = 4) :
  a + b + c + d = 12 := by
  sorry

end sum_of_integers_l744_74437


namespace range_of_a_l744_74493

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 2)*x + 1 ≥ 0) → 0 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l744_74493


namespace quadratic_inverse_unique_solution_l744_74449

/-- A quadratic function with its inverse -/
structure QuadraticWithInverse where
  a : ℝ
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  f_inv : ℝ → ℝ
  h_f : ∀ x, f x = a * x^2 + b * x + c
  h_f_inv : ∀ x, f_inv x = c * x^2 + b * x + a
  h_inverse : (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x)

/-- Theorem stating the unique solution for a, b, and c -/
theorem quadratic_inverse_unique_solution (q : QuadraticWithInverse) :
  q.a = -1 ∧ q.b = 1 ∧ q.c = 0 := by
  sorry

end quadratic_inverse_unique_solution_l744_74449


namespace initial_water_percentage_l744_74498

theorem initial_water_percentage (
  initial_volume : ℝ) 
  (kola_percentage : ℝ)
  (added_sugar : ℝ) 
  (added_water : ℝ) 
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ) :
  initial_volume = 340 →
  kola_percentage = 6 →
  added_sugar = 3.2 →
  added_water = 10 →
  added_kola = 6.8 →
  final_sugar_percentage = 14.111111111111112 →
  ∃ initial_water_percentage : ℝ,
    initial_water_percentage = 80 ∧
    initial_water_percentage + kola_percentage + (100 - initial_water_percentage - kola_percentage) = 100 ∧
    (((100 - initial_water_percentage - kola_percentage) / 100 * initial_volume + added_sugar) / 
      (initial_volume + added_sugar + added_water + added_kola)) * 100 = final_sugar_percentage :=
by sorry

end initial_water_percentage_l744_74498


namespace joes_kids_haircuts_l744_74453

-- Define the time it takes for each type of haircut
def womens_haircut_time : ℕ := 50
def mens_haircut_time : ℕ := 15
def kids_haircut_time : ℕ := 25

-- Define the number of women's and men's haircuts
def num_womens_haircuts : ℕ := 3
def num_mens_haircuts : ℕ := 2

-- Define the total time spent cutting hair
def total_time : ℕ := 255

-- Define a function to calculate the number of kids' haircuts
def num_kids_haircuts (w m k : ℕ) : ℕ :=
  (total_time - (w * womens_haircut_time + m * mens_haircut_time)) / k

-- Theorem statement
theorem joes_kids_haircuts :
  num_kids_haircuts num_womens_haircuts num_mens_haircuts kids_haircut_time = 3 := by
  sorry

end joes_kids_haircuts_l744_74453


namespace rectangle_ratio_l744_74476

/-- Configuration of rectangles around a square -/
structure RectangleConfig where
  /-- Side length of the smaller square -/
  s : ℝ
  /-- Shorter side of the rectangle -/
  y : ℝ
  /-- Longer side of the rectangle -/
  x : ℝ
  /-- The side length of the smaller square is 1 -/
  h1 : s = 1
  /-- The side length of the larger square is s + 2y -/
  h2 : s + 2*y = 2*s
  /-- The side length of the larger square is also x + s -/
  h3 : x + s = 2*s

/-- The ratio of the longer side to the shorter side of the rectangle is 2 -/
theorem rectangle_ratio (config : RectangleConfig) : x / y = 2 :=
by sorry

end rectangle_ratio_l744_74476


namespace population_after_two_years_l744_74469

def initial_population : ℕ := 10000
def first_year_rate : ℚ := 1.05
def second_year_rate : ℚ := 0.95

theorem population_after_two_years :
  (↑initial_population * first_year_rate * second_year_rate).floor = 9975 := by
  sorry

end population_after_two_years_l744_74469


namespace min_x_prime_factorization_l744_74419

theorem min_x_prime_factorization (x y : ℕ+) (a b : ℕ) (c d : ℕ) 
  (h1 : 4 * x^7 = 13 * y^17)
  (h2 : x = a^c * b^d)
  (h3 : Nat.Prime a)
  (h4 : Nat.Prime b)
  (h5 : ∀ (w z : ℕ+) (e f : ℕ) (p q : ℕ), 
        4 * w^7 = 13 * z^17 → 
        w = p^e * q^f → 
        Nat.Prime p → 
        Nat.Prime q → 
        w ≤ x) : 
  a + b + c + d = 19 := by
sorry

end min_x_prime_factorization_l744_74419


namespace remainder_theorem_l744_74467

theorem remainder_theorem (n m p : ℤ) 
  (hn : n % 18 = 10)
  (hm : m % 27 = 16)
  (hp : p % 6 = 4) :
  (2*n + 3*m - p) % 9 = 1 := by
  sorry

end remainder_theorem_l744_74467


namespace local_minimum_implies_b_range_l744_74465

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- State the theorem
theorem local_minimum_implies_b_range :
  ∀ b : ℝ, (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ IsLocalMin (f b) x) → 0 < b ∧ b < 1 :=
by sorry

end local_minimum_implies_b_range_l744_74465


namespace A_intersect_B_eq_unit_interval_l744_74420

-- Define set A
def A : Set ℝ := {x | x + 1 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (1 - 2^x)}

-- Theorem stating that the intersection of A and B is [0, 1)
theorem A_intersect_B_eq_unit_interval :
  A ∩ B = Set.Icc 0 1 := by sorry

end A_intersect_B_eq_unit_interval_l744_74420


namespace unique_three_digit_number_l744_74411

/-- A three-digit number is represented as 100 * a + 10 * b + c, where a, b, c are single digits -/
def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

/-- The number is 12 times the sum of its digits -/
def twelve_times_sum (a b c : ℕ) : Prop :=
  100 * a + 10 * b + c = 12 * (a + b + c)

theorem unique_three_digit_number :
  ∃! (a b c : ℕ), three_digit_number a b c ∧ twelve_times_sum a b c ∧
    100 * a + 10 * b + c = 108 :=
by sorry

end unique_three_digit_number_l744_74411


namespace min_value_sum_squares_l744_74439

theorem min_value_sum_squares (x y z a : ℝ) (h : x + 2*y + 3*z = a) :
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end min_value_sum_squares_l744_74439


namespace intersection_A_B_l744_74458

def A : Set ℝ := {x | -3 ≤ 2*x - 1 ∧ 2*x - 1 < 3}
def B : Set ℝ := {x | ∃ k : ℤ, x = 2*k + 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := by
  sorry

end intersection_A_B_l744_74458


namespace distance_between_points_l744_74457

/-- The distance between two points when two people walk towards each other --/
theorem distance_between_points (speed_a speed_b : ℝ) (midpoint_offset : ℝ) : 
  speed_a = 70 →
  speed_b = 60 →
  midpoint_offset = 80 →
  (speed_a - speed_b) * ((2 * midpoint_offset) / (speed_a - speed_b)) = speed_a + speed_b →
  (speed_a + speed_b) * ((2 * midpoint_offset) / (speed_a - speed_b)) = 2080 :=
by
  sorry

#check distance_between_points

end distance_between_points_l744_74457


namespace six_people_three_events_outcomes_l744_74477

/-- The number of possible outcomes for champions in a competition. -/
def championOutcomes (people : ℕ) (events : ℕ) : ℕ :=
  people ^ events

/-- Theorem stating the number of possible outcomes for 6 people in 3 events. -/
theorem six_people_three_events_outcomes :
  championOutcomes 6 3 = 216 := by
  sorry

end six_people_three_events_outcomes_l744_74477


namespace contrapositive_equivalence_l744_74478

theorem contrapositive_equivalence (m : ℝ) : 
  (¬(∃ x : ℝ, x^2 = m) → m < 0) ↔ 
  (m ≥ 0 → ∃ x : ℝ, x^2 = m) := by sorry

end contrapositive_equivalence_l744_74478


namespace barbara_candies_left_l744_74487

/-- The number of candies Barbara has left after using some -/
def candies_left (initial : Float) (used : Float) : Float :=
  initial - used

/-- Theorem: If Barbara initially has 18.0 candies and uses 9.0 candies,
    then the number of candies she has left is 9.0. -/
theorem barbara_candies_left :
  candies_left 18.0 9.0 = 9.0 := by
  sorry

end barbara_candies_left_l744_74487


namespace smallest_cube_box_for_pyramid_l744_74496

theorem smallest_cube_box_for_pyramid (pyramid_height base_length base_width : ℝ) 
  (h_height : pyramid_height = 15)
  (h_base_length : base_length = 9)
  (h_base_width : base_width = 12) :
  let box_side := max pyramid_height (max base_length base_width)
  (box_side ^ 3 : ℝ) = 3375 :=
by sorry

end smallest_cube_box_for_pyramid_l744_74496


namespace min_cos_C_in_triangle_l744_74424

theorem min_cos_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (side_relation : a^2 + b^2 = (5/2) * c^2) : 
  ∃ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2*a*b) ∧ cos_C ≥ 3/5 :=
sorry

end min_cos_C_in_triangle_l744_74424


namespace frequency_converges_to_probability_l744_74452

-- Define a random experiment
def RandomExperiment : Type := Unit

-- Define an event in the experiment
def Event (e : RandomExperiment) : Type := Unit

-- Define the probability of an event
def probability (e : RandomExperiment) (A : Event e) : ℝ := sorry

-- Define the frequency of an event after n trials
def frequency (e : RandomExperiment) (A : Event e) (n : ℕ) : ℝ := sorry

-- Theorem: As the number of trials approaches infinity, 
-- the frequency converges to the probability
theorem frequency_converges_to_probability 
  (e : RandomExperiment) (A : Event e) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
  |frequency e A n - probability e A| < ε :=
sorry

end frequency_converges_to_probability_l744_74452


namespace quadratic_zero_discriminant_geometric_progression_l744_74488

/-- 
Given a quadratic equation ax^2 + 6bx + 9c = 0 with zero discriminant,
prove that a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) 
  (h_quad : ∀ x, a * x^2 + 6 * b * x + 9 * c = 0)
  (h_discr : (6 * b)^2 - 4 * a * (9 * c) = 0) :
  ∃ r : ℝ, b = a * r ∧ c = b * r :=
sorry

end quadratic_zero_discriminant_geometric_progression_l744_74488


namespace sum_of_cubes_divisible_by_three_l744_74409

theorem sum_of_cubes_divisible_by_three (n : ℤ) : 
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 3 * k := by
sorry

end sum_of_cubes_divisible_by_three_l744_74409


namespace cosine_value_in_triangle_l744_74400

/-- In a triangle ABC, if b cos C = (3a - c) cos B, then cos B = 1/3 -/
theorem cosine_value_in_triangle (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * Real.cos C = (3 * a - c) * Real.cos B) →
  Real.cos B = 1 / 3 := by
sorry


end cosine_value_in_triangle_l744_74400


namespace system_solution_l744_74490

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k + 1 →
  2*x + y = 1 →
  x + y = 3 →
  k = 7 := by
sorry

end system_solution_l744_74490


namespace cheap_gym_cost_l744_74473

/-- Represents the monthly cost of gym memberships and related calculations -/
def gym_costs (cheap_monthly : ℝ) : Prop :=
  let expensive_monthly := 3 * cheap_monthly
  let cheap_signup := 50
  let expensive_signup := 4 * expensive_monthly
  let cheap_yearly := cheap_signup + 12 * cheap_monthly
  let expensive_yearly := expensive_signup + 12 * expensive_monthly
  cheap_yearly + expensive_yearly = 650

theorem cheap_gym_cost : ∃ (cheap_monthly : ℝ), gym_costs cheap_monthly ∧ cheap_monthly = 10 := by
  sorry

end cheap_gym_cost_l744_74473


namespace handshake_theorem_l744_74430

theorem handshake_theorem (n : ℕ) (h : n = 40) : 
  (n * (n - 1)) / 2 = 780 := by
  sorry

#check handshake_theorem

end handshake_theorem_l744_74430


namespace coupon_one_best_l744_74460

/-- Represents the discount amount for a given coupon and price -/
def discount (coupon : Nat) (price : ℝ) : ℝ :=
  match coupon with
  | 1 => 0.15 * price
  | 2 => 30
  | 3 => 0.25 * (price - 120)
  | _ => 0

/-- Theorem stating when coupon 1 is the best choice -/
theorem coupon_one_best (price : ℝ) :
  (∀ c : Nat, c ≠ 1 → discount 1 price > discount c price) ↔ 200 < price ∧ price < 300 := by
  sorry


end coupon_one_best_l744_74460


namespace sufficient_not_necessary_condition_l744_74474

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (b < -4 → ∀ a, |a| + |b| > 4) ∧ 
  (∃ a b, |a| + |b| > 4 ∧ b ≥ -4) := by
sorry

end sufficient_not_necessary_condition_l744_74474


namespace total_complaints_over_five_days_l744_74435

/-- Represents the different staff shortage scenarios -/
inductive StaffShortage
  | Normal
  | TwentyPercent
  | FortyPercent

/-- Represents the different self-checkout states -/
inductive SelfCheckout
  | Working
  | PartiallyBroken
  | CompletelyBroken

/-- Represents the different weather conditions -/
inductive Weather
  | Clear
  | Rainy
  | Snowstorm

/-- Represents the different special events -/
inductive SpecialEvent
  | Normal
  | Holiday
  | OngoingSale

/-- Represents the conditions for a single day -/
structure DayConditions where
  staffShortage : StaffShortage
  selfCheckout : SelfCheckout
  weather : Weather
  specialEvent : SpecialEvent

/-- Calculates the number of complaints for a given day based on its conditions -/
def calculateComplaints (baseComplaints : ℕ) (conditions : DayConditions) : ℕ :=
  sorry

/-- The base number of complaints per day -/
def baseComplaints : ℕ := 120

/-- The conditions for each of the five days -/
def dayConditions : List DayConditions := [
  { staffShortage := StaffShortage.TwentyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Rainy, specialEvent := SpecialEvent.OngoingSale },
  { staffShortage := StaffShortage.FortyPercent, selfCheckout := SelfCheckout.PartiallyBroken, weather := Weather.Clear, specialEvent := SpecialEvent.Holiday },
  { staffShortage := StaffShortage.FortyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Snowstorm, specialEvent := SpecialEvent.Normal },
  { staffShortage := StaffShortage.Normal, selfCheckout := SelfCheckout.Working, weather := Weather.Rainy, specialEvent := SpecialEvent.OngoingSale },
  { staffShortage := StaffShortage.TwentyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Clear, specialEvent := SpecialEvent.Holiday }
]

/-- Theorem stating that the total number of complaints over the five days is 1038 -/
theorem total_complaints_over_five_days :
  (dayConditions.map (calculateComplaints baseComplaints)).sum = 1038 := by
  sorry

end total_complaints_over_five_days_l744_74435


namespace angle5_is_36_degrees_l744_74402

-- Define the angles
variable (angle1 angle2 angle5 : ℝ)

-- Define the conditions
axiom parallel_lines : True  -- m ∥ n
axiom angle1_is_quarter_angle2 : angle1 = (1 / 4) * angle2
axiom alternate_interior_angles : angle5 = angle1
axiom straight_line : angle2 + angle5 = 180

-- Theorem to prove
theorem angle5_is_36_degrees : angle5 = 36 := by
  sorry

end angle5_is_36_degrees_l744_74402


namespace chloe_min_score_l744_74494

/-- The minimum score needed on the fifth test to achieve a given average -/
def min_score_for_average (test1 test2 test3 test4 : ℚ) (required_avg : ℚ) : ℚ :=
  5 * required_avg - (test1 + test2 + test3 + test4)

/-- Proof that Chloe needs at least 86% on her fifth test -/
theorem chloe_min_score :
  let test1 : ℚ := 84
  let test2 : ℚ := 87
  let test3 : ℚ := 78
  let test4 : ℚ := 90
  let required_avg : ℚ := 85
  min_score_for_average test1 test2 test3 test4 required_avg = 86 := by
  sorry

#eval min_score_for_average 84 87 78 90 85

end chloe_min_score_l744_74494


namespace interest_problem_l744_74412

theorem interest_problem (P : ℝ) : 
  (P * 0.04 + P * 0.06 + P * 0.08 = 2700) → P = 15000 := by
  sorry

end interest_problem_l744_74412


namespace circumcircle_of_triangle_ABC_l744_74422

/-- The circumcircle of a triangle is the circle that passes through all three vertices of the triangle. -/
def is_circumcircle (a b c : ℝ × ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  f a.1 a.2 = 0 ∧ f b.1 b.2 = 0 ∧ f c.1 c.2 = 0

/-- The equation of a circle in general form is x^2 + y^2 + Dx + Ey + F = 0 -/
def circle_equation (D E F : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + D*x + E*y + F

theorem circumcircle_of_triangle_ABC :
  let A : ℝ × ℝ := (-1, 5)
  let B : ℝ × ℝ := (5, 5)
  let C : ℝ × ℝ := (6, -2)
  let f (x y : ℝ) := circle_equation (-4) (-2) (-20) x y
  is_circumcircle A B C f :=
sorry

end circumcircle_of_triangle_ABC_l744_74422


namespace air_quality_probability_l744_74444

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) 
  (h1 : p_good = 0.8) 
  (h2 : p_consecutive = 0.68) : 
  p_consecutive / p_good = 0.85 := by
  sorry

end air_quality_probability_l744_74444


namespace floor_x_length_l744_74436

/-- Represents the dimensions of a rectangular floor -/
structure Floor where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular floor -/
def area (f : Floor) : ℝ := f.width * f.length

theorem floor_x_length
  (x y : Floor)
  (h1 : area x = area y)
  (h2 : x.width = 10)
  (h3 : y.width = 9)
  (h4 : y.length = 20) :
  x.length = 18 := by
  sorry

end floor_x_length_l744_74436


namespace largest_fraction_l744_74425

theorem largest_fraction : 
  let f1 := 5 / 11
  let f2 := 6 / 13
  let f3 := 19 / 39
  let f4 := 101 / 203
  let f5 := 152 / 303
  let f6 := 80 / 159
  (f6 > f1) ∧ (f6 > f2) ∧ (f6 > f3) ∧ (f6 > f4) ∧ (f6 > f5) := by
  sorry

end largest_fraction_l744_74425


namespace fruit_weights_l744_74416

/-- Represents the fruits on the table -/
inductive Fruit
| orange
| banana
| mandarin
| peach
| apple

/-- Assigns weights to fruits -/
def weight : Fruit → ℕ
| Fruit.orange => 280
| Fruit.banana => 170
| Fruit.mandarin => 100
| Fruit.peach => 200
| Fruit.apple => 150

/-- The set of all possible weights -/
def weights : Set ℕ := {100, 150, 170, 200, 280}

theorem fruit_weights :
  (∀ f : Fruit, weight f ∈ weights) ∧
  (weight Fruit.peach < weight Fruit.orange) ∧
  (weight Fruit.apple < weight Fruit.banana) ∧
  (weight Fruit.banana < weight Fruit.peach) ∧
  (weight Fruit.mandarin < weight Fruit.banana) ∧
  (weight Fruit.apple + weight Fruit.banana > weight Fruit.orange) ∧
  (∀ w : ℕ, w ∈ weights → ∃! f : Fruit, weight f = w) :=
by sorry

end fruit_weights_l744_74416


namespace appropriate_units_l744_74466

-- Define the mass units
inductive MassUnit
| Kilogram
| Gram
| Ton

-- Define a structure for an object with a weight and unit
structure WeightedObject where
  weight : ℝ
  unit : MassUnit

-- Define the objects
def basketOfEggs : WeightedObject := { weight := 5, unit := MassUnit.Kilogram }
def honeybee : WeightedObject := { weight := 5, unit := MassUnit.Gram }
def tank : WeightedObject := { weight := 6, unit := MassUnit.Ton }

-- Function to determine if a unit is appropriate for an object
def isAppropriateUnit (obj : WeightedObject) : Prop :=
  match obj with
  | { weight := w, unit := MassUnit.Kilogram } => w ≥ 1 ∧ w < 1000
  | { weight := w, unit := MassUnit.Gram } => w ≥ 0.1 ∧ w < 1000
  | { weight := w, unit := MassUnit.Ton } => w ≥ 1 ∧ w < 1000

-- Theorem stating that the given units are appropriate for each object
theorem appropriate_units :
  isAppropriateUnit basketOfEggs ∧
  isAppropriateUnit honeybee ∧
  isAppropriateUnit tank := by
  sorry

end appropriate_units_l744_74466


namespace prob_first_success_third_trial_l744_74497

/-- Probability of first success on third trial in a geometric distribution -/
theorem prob_first_success_third_trial (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  let q := 1 - p
  (q ^ 2) * p = p * (1 - p) ^ 2 :=
by sorry

end prob_first_success_third_trial_l744_74497


namespace quadratic_and_inequality_solution_l744_74429

theorem quadratic_and_inequality_solution :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    (x₁^2 - 2*x₁ - 4 = 0) ∧ (x₂^2 - 2*x₂ - 4 = 0)) ∧
  (∀ x : ℝ, (2*(x-1) ≥ -4 ∧ (3*x-6)/2 < x-1) ↔ (-1 ≤ x ∧ x < 4)) :=
by sorry

end quadratic_and_inequality_solution_l744_74429


namespace quadratic_inequality_range_l744_74492

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ 
  (a > -2 ∧ a ≤ 2) :=
by sorry

end quadratic_inequality_range_l744_74492


namespace system_solution_l744_74461

/-- Given a system of equations x * y = a and x^5 + y^5 = b^5, this theorem states the solutions
    for different cases of a and b. -/
theorem system_solution (a b : ℝ) :
  (∀ x y : ℝ, x * y = a ∧ x^5 + y^5 = b^5 →
    (a = 0 ∧ b = 0 ∧ ∃ t : ℝ, x = t ∧ y = -t) ∨
    ((16 * b^5 ≤ a^5 ∧ a^5 < 0) ∨ (0 < a^5 ∧ a^5 ≤ 16 * b^5) ∧
      ((x = a/2 + Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4) ∧
        y = a/2 - Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4)) ∨
       (x = a/2 - Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4) ∧
        y = a/2 + Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4))))) :=
by sorry

end system_solution_l744_74461


namespace binomial_expansion_coefficient_sum_l744_74406

/-- The sum of the coefficients of the last three terms in the binomial expansion -/
def sum_of_last_three_coefficients (n : ℕ) : ℕ := 1 + n + n * (n - 1) / 2

/-- The theorem stating that if the sum of the coefficients of the last three terms 
    in the expansion of (√x + 2/√x)^n is 79, then n = 12 -/
theorem binomial_expansion_coefficient_sum (n : ℕ) : 
  sum_of_last_three_coefficients n = 79 → n = 12 := by
  sorry

end binomial_expansion_coefficient_sum_l744_74406


namespace simultaneous_inequalities_l744_74426

theorem simultaneous_inequalities (a b : ℝ) :
  (a < b ∧ 1 / a < 1 / b) ↔ a < 0 ∧ 0 < b := by
  sorry

end simultaneous_inequalities_l744_74426


namespace quadratic_equations_solutions_l744_74462

theorem quadratic_equations_solutions :
  (∃ (s : Set ℝ), s = {x : ℝ | 2 * x^2 - x = 0} ∧ s = {0, 1/2}) ∧
  (∃ (t : Set ℝ), t = {x : ℝ | (2 * x + 1)^2 - 9 = 0} ∧ t = {1, -2}) := by
sorry

end quadratic_equations_solutions_l744_74462


namespace constant_term_expansion_l744_74472

/-- The constant term in the expansion of (x + 2/x)^6 -/
def constant_term : ℕ := 160

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_expansion :
  constant_term = binomial 6 3 * 2^3 := by sorry

end constant_term_expansion_l744_74472


namespace larger_number_proof_l744_74455

/-- Given two positive integers with specific HCF and LCM conditions, prove the larger one is 230 -/
theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 23) 
  (h2 : Nat.lcm a b = 23 * 9 * 10) (h3 : a > b) : a = 230 := by
  sorry

end larger_number_proof_l744_74455


namespace greg_earnings_l744_74428

/-- Represents the rates and walking details for Greg's dog walking business -/
structure DogWalkingRates where
  small_base : ℝ := 15
  small_per_minute : ℝ := 1
  medium_base : ℝ := 20
  medium_per_minute : ℝ := 1.25
  large_base : ℝ := 25
  large_per_minute : ℝ := 1.5
  small_dogs : ℕ := 3
  small_minutes : ℕ := 12
  medium_dogs : ℕ := 2
  medium_minutes : ℕ := 18
  large_dogs : ℕ := 1
  large_minutes : ℕ := 25

/-- Calculates Greg's total earnings from dog walking -/
def calculateEarnings (rates : DogWalkingRates) : ℝ :=
  (rates.small_base * rates.small_dogs + rates.small_per_minute * rates.small_dogs * rates.small_minutes) +
  (rates.medium_base * rates.medium_dogs + rates.medium_per_minute * rates.medium_dogs * rates.medium_minutes) +
  (rates.large_base * rates.large_dogs + rates.large_per_minute * rates.large_dogs * rates.large_minutes)

/-- Theorem stating that Greg's total earnings are $228.50 -/
theorem greg_earnings (rates : DogWalkingRates) : calculateEarnings rates = 228.5 := by
  sorry

end greg_earnings_l744_74428


namespace expression_simplification_l744_74421

theorem expression_simplification (x : ℝ) : 
  ((((x + 2)^2 * (x^2 - 2*x + 2)^2) / (x^3 + 2)^2)^2 * 
   (((x - 2)^2 * (x^2 + 2*x + 2)^2) / (x^3 - 2)^2)^2) = 1 := by
  sorry

end expression_simplification_l744_74421


namespace imaginary_part_of_z_l744_74441

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : 
  Complex.im z = -1 := by sorry

end imaginary_part_of_z_l744_74441


namespace senior_ticket_cost_l744_74405

theorem senior_ticket_cost 
  (total_tickets : ℕ) 
  (adult_price : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) 
  (h2 : adult_price = 21) 
  (h3 : total_receipts = 8748) 
  (h4 : senior_tickets = 327) :
  ∃ (senior_price : ℕ), 
    senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) = total_receipts ∧ 
    senior_price = 15 := by
sorry

end senior_ticket_cost_l744_74405


namespace unique_solution_quartic_equation_l744_74414

theorem unique_solution_quartic_equation :
  ∃! x : ℝ, x^4 + (2 - x)^4 + 2*x = 34 := by
  sorry

end unique_solution_quartic_equation_l744_74414


namespace grocery_store_inventory_l744_74483

theorem grocery_store_inventory (regular_soda diet_soda apples : ℕ) 
  (h1 : regular_soda = 72)
  (h2 : diet_soda = 32)
  (h3 : apples = 78) :
  regular_soda + diet_soda - apples = 26 := by
  sorry

end grocery_store_inventory_l744_74483


namespace angle_B_measure_side_b_value_l744_74408

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Part 1
theorem angle_B_measure (t : Triangle) (h : (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)) :
  t.B = 2 * π / 3 := by sorry

-- Part 2
theorem side_b_value (t : Triangle) (h1 : t.a = 4) (h2 : t.S = 5 * Real.sqrt 3) (h3 : t.B = 2 * π / 3) :
  t.b = Real.sqrt 61 := by sorry

end angle_B_measure_side_b_value_l744_74408


namespace internet_cost_proof_l744_74471

/-- The daily cost of internet service -/
def daily_cost : ℝ := 0.28

/-- The number of days covered by the payment -/
def days_covered : ℕ := 25

/-- The amount paid -/
def payment : ℝ := 7

/-- The maximum allowed debt -/
def max_debt : ℝ := 5

theorem internet_cost_proof :
  daily_cost * days_covered = payment ∧
  daily_cost * (days_covered + 1) > payment + max_debt :=
by sorry

end internet_cost_proof_l744_74471


namespace bill_throws_21_objects_l744_74434

/-- The number of sticks Ted throws -/
def ted_sticks : ℕ := 10

/-- The number of rocks Ted throws -/
def ted_rocks : ℕ := 10

/-- The number of sticks Bill throws -/
def bill_sticks : ℕ := ted_sticks + 6

/-- The number of rocks Bill throws -/
def bill_rocks : ℕ := ted_rocks / 2

/-- The total number of objects Bill throws -/
def bill_total : ℕ := bill_sticks + bill_rocks

theorem bill_throws_21_objects : bill_total = 21 := by
  sorry

end bill_throws_21_objects_l744_74434


namespace alloy_price_example_l744_74417

/-- The price of the alloy per kg when two metals are mixed in equal proportions -/
def alloy_price (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / 2

/-- Theorem: The price of an alloy made from two metals costing 68 and 96 per kg, 
    mixed in equal proportions, is 82 per kg -/
theorem alloy_price_example : alloy_price 68 96 = 82 := by
  sorry

end alloy_price_example_l744_74417


namespace pipe_fill_time_l744_74499

/-- Represents the time (in hours) it takes for a pipe to fill a tank without a leak -/
def fill_time : ℝ := 6

/-- Represents the time (in hours) it takes for the pipe to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 8

/-- Represents the time (in hours) it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 24

/-- Proves that the time it takes for the pipe to fill the tank without the leak is 6 hours -/
theorem pipe_fill_time : 
  (1 / fill_time - 1 / leak_empty_time) * fill_time_with_leak = 1 :=
sorry

end pipe_fill_time_l744_74499


namespace pizza_piece_volume_l744_74495

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/3 →
  diameter = 18 →
  num_pieces = 18 →
  (π * (diameter/2)^2 * thickness) / num_pieces = 3*π/2 := by
  sorry

end pizza_piece_volume_l744_74495


namespace complex_fraction_squared_l744_74445

theorem complex_fraction_squared (i : ℂ) (hi : i^2 = -1) :
  ((1 - i) / (1 + i))^2 = -1 := by sorry

end complex_fraction_squared_l744_74445


namespace b_value_proof_l744_74463

theorem b_value_proof (b : ℚ) (h : b + b/4 = 5/2) : b = 2 := by
  sorry

end b_value_proof_l744_74463


namespace certain_number_value_l744_74427

theorem certain_number_value : ∃ x : ℝ, (35 / 100) * x = (20 / 100) * 700 ∧ x = 400 := by
  sorry

end certain_number_value_l744_74427


namespace equal_coin_count_l744_74481

/-- Represents the value of each coin type in cents -/
def coin_value : Fin 5 → ℕ
  | 0 => 1    -- penny
  | 1 => 5    -- nickel
  | 2 => 10   -- dime
  | 3 => 25   -- quarter
  | 4 => 50   -- half dollar

/-- The theorem statement -/
theorem equal_coin_count (x : ℕ) (h : x * (coin_value 0 + coin_value 1 + coin_value 2 + coin_value 3 + coin_value 4) = 273) :
  5 * x = 15 := by
  sorry

#check equal_coin_count

end equal_coin_count_l744_74481


namespace class_distribution_l744_74410

theorem class_distribution (total : ℕ) (girls boys_carrots boys_apples : ℕ) : 
  total = 33 → 
  girls + boys_carrots + boys_apples = total →
  3 * boys_carrots + boys_apples = girls →
  boys_apples = girls →
  4 * boys_carrots = girls →
  girls = 15 ∧ boys_carrots = 6 ∧ boys_apples = 12 := by
  sorry

end class_distribution_l744_74410


namespace six_player_four_games_tournament_l744_74464

/-- Represents a chess tournament --/
structure ChessTournament where
  numPlayers : Nat
  gamesPerPlayer : Nat

/-- Calculates the total number of games in a chess tournament --/
def totalGames (t : ChessTournament) : Nat :=
  (t.numPlayers * t.gamesPerPlayer) / 2

/-- Theorem: In a tournament with 6 players where each plays 4 others, there are 10 games total --/
theorem six_player_four_games_tournament :
  ∀ (t : ChessTournament),
    t.numPlayers = 6 →
    t.gamesPerPlayer = 4 →
    totalGames t = 10 := by
  sorry

#check six_player_four_games_tournament

end six_player_four_games_tournament_l744_74464


namespace patty_weight_factor_l744_74479

/-- Given:
  - Robbie weighs 100 pounds
  - Patty was initially x times as heavy as Robbie
  - Patty lost 235 pounds
  - After weight loss, Patty weighs 115 pounds more than Robbie
Prove that x = 4.5 -/
theorem patty_weight_factor (x : ℝ) 
  (robbie_weight : ℝ) (patty_weight_loss : ℝ) (patty_final_difference : ℝ)
  (h1 : robbie_weight = 100)
  (h2 : patty_weight_loss = 235)
  (h3 : patty_final_difference = 115)
  (h4 : x * robbie_weight - patty_weight_loss = robbie_weight + patty_final_difference) :
  x = 4.5 := by
sorry

end patty_weight_factor_l744_74479


namespace minimum_bottles_needed_l744_74447

def small_bottle_capacity : ℕ := 45
def large_bottle_capacity : ℕ := 600
def already_filled : ℕ := 90

theorem minimum_bottles_needed : 
  ∃ (n : ℕ), n * small_bottle_capacity + already_filled ≥ large_bottle_capacity ∧ 
  ∀ (m : ℕ), m * small_bottle_capacity + already_filled ≥ large_bottle_capacity → n ≤ m :=
by
  sorry

end minimum_bottles_needed_l744_74447


namespace expression_value_at_five_l744_74431

theorem expression_value_at_five :
  let x : ℚ := 5
  (x^2 + x - 12) / (x - 4) = 18 :=
by sorry

end expression_value_at_five_l744_74431


namespace triangle_cosine_theorem_l744_74403

theorem triangle_cosine_theorem (A : ℝ) (a m : ℝ) (θ : ℝ) : 
  A = 24 →
  a = 12 →
  m = 5 →
  A = (1/2) * a * m * Real.sin θ →
  Real.cos θ = 3/5 := by
  sorry

end triangle_cosine_theorem_l744_74403


namespace vector_problem_l744_74432

/-- Given vectors in R^2 -/
def OA : Fin 2 → ℝ := ![3, -4]
def OB : Fin 2 → ℝ := ![6, -3]
def OC (m : ℝ) : Fin 2 → ℝ := ![5 - m, -3 - m]

/-- A, B, and C are collinear if and only if the cross product of AB and AC is zero -/
def collinear (m : ℝ) : Prop :=
  let AB := OB - OA
  let AC := OC m - OA
  AB 0 * AC 1 = AB 1 * AC 0

/-- ABC is a right-angled triangle if and only if one of its angles is 90 degrees -/
def right_angled (m : ℝ) : Prop :=
  let AB := OB - OA
  let BC := OC m - OB
  let AC := OC m - OA
  AB • BC = 0 ∨ BC • AC = 0 ∨ AC • AB = 0

/-- Main theorem -/
theorem vector_problem (m : ℝ) :
  (collinear m → m = 1/2) ∧
  (right_angled m → m = 7/4 ∨ m = -3/4 ∨ m = (1 + Real.sqrt 5)/2 ∨ m = (1 - Real.sqrt 5)/2) :=
sorry

end vector_problem_l744_74432


namespace blue_balls_count_l744_74407

theorem blue_balls_count (red_balls : ℕ) (blue_balls : ℕ) 
  (h1 : red_balls = 25)
  (h2 : red_balls = 2 * blue_balls + 3) :
  blue_balls = 11 := by
  sorry

end blue_balls_count_l744_74407


namespace max_m_value_l744_74442

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2*x|

theorem max_m_value (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) 
  (h4 : f a = 3 * f b) :
  ∃ m : ℤ, (∀ n : ℤ, a^2 + b^2 > ↑n → n ≤ m) ∧ a^2 + b^2 > ↑m :=
by sorry

end max_m_value_l744_74442


namespace polynomial_division_l744_74470

theorem polynomial_division (x y : ℝ) (h : y ≠ 0) : (3 * x * y + y) / y = 3 * x + 1 := by
  sorry

end polynomial_division_l744_74470


namespace purely_imaginary_complex_number_l744_74454

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a + 1) = (a^2 - 2*a - 3) + Complex.I * (a + 1)) → a = 3 := by
  sorry

end purely_imaginary_complex_number_l744_74454


namespace two_zeros_iff_k_range_l744_74468

/-- The function f(x) = xe^x - k has exactly two zeros if and only if -1/e < k < 0 -/
theorem two_zeros_iff_k_range (k : ℝ) :
  (∃! (a b : ℝ), a ≠ b ∧ a * Real.exp a - k = 0 ∧ b * Real.exp b - k = 0) ↔
  -1 / Real.exp 1 < k ∧ k < 0 := by sorry

end two_zeros_iff_k_range_l744_74468


namespace hyperbola_asymptotes_l744_74475

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2/3 * x ∨ y = -2/3 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(2/3)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l744_74475


namespace sum_of_solutions_is_pi_over_six_l744_74423

theorem sum_of_solutions_is_pi_over_six :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π ∧
  (1 / Real.sin x) + (1 / Real.cos x) = 2 * Real.sqrt 3 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π ∧
    (1 / Real.sin y) + (1 / Real.cos y) = 2 * Real.sqrt 3 →
    y = x) ∧
  x = π / 6 :=
by sorry

end sum_of_solutions_is_pi_over_six_l744_74423


namespace yellow_ball_count_l744_74486

theorem yellow_ball_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob_red : ℚ) : 
  red = 9 →
  yellow + red = total →
  prob_red = 1/3 →
  prob_red = red / total →
  yellow = 18 :=
sorry

end yellow_ball_count_l744_74486


namespace triangle_properties_l744_74438

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions here
  c = Real.sqrt 2 ∧
  Real.cos C = 3/4 ∧
  2 * c * Real.sin A = b * Real.sin C

-- State the theorem
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  b = 2 ∧ 
  Real.sin A = Real.sqrt 14 / 8 ∧ 
  Real.sin (2 * A + π/3) = (5 * Real.sqrt 7 + 9 * Real.sqrt 3) / 32 := by
  sorry


end triangle_properties_l744_74438


namespace remaining_time_formula_l744_74415

/-- Represents the exam scenario for Jessica -/
structure ExamScenario where
  totalTime : ℕ  -- Total time for the exam in minutes
  totalQuestions : ℕ  -- Total number of questions
  answeredQuestions : ℕ  -- Number of questions answered so far
  timeUsed : ℕ  -- Time used so far in minutes
  penaltyPerIncorrect : ℕ  -- Time penalty for each incorrect answer in minutes

/-- Calculates the remaining time after penalties -/
def remainingTimeAfterPenalties (scenario : ExamScenario) (incorrectAnswers : ℕ) : ℤ :=
  scenario.totalTime - scenario.timeUsed - 
  (scenario.totalQuestions - scenario.answeredQuestions) * 
  (scenario.timeUsed / scenario.answeredQuestions) -
  incorrectAnswers * scenario.penaltyPerIncorrect

/-- Theorem stating that the remaining time after penalties is 15 - 2x -/
theorem remaining_time_formula (incorrectAnswers : ℕ) : 
  remainingTimeAfterPenalties 
    { totalTime := 90
    , totalQuestions := 100
    , answeredQuestions := 20
    , timeUsed := 15
    , penaltyPerIncorrect := 2 } 
    incorrectAnswers = 15 - 2 * incorrectAnswers :=
by sorry

end remaining_time_formula_l744_74415


namespace range_of_a_l744_74440

def A (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}
def B : Set ℝ := {x | x ≤ 3 ∨ x > 5}
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

def p (a : ℝ) : Prop := A a ⊆ B
def q (a : ℝ) : Prop := ∀ x > (1/2 : ℝ), Monotone (f a)

theorem range_of_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((1/2 < a ∧ a ≤ 2) ∨ a > 4) :=
sorry

end range_of_a_l744_74440


namespace min_garden_cost_l744_74448

-- Define the regions and their areas
def region1_area : ℕ := 10  -- 5x2
def region2_area : ℕ := 9   -- 3x3
def region3_area : ℕ := 20  -- 5x4
def region4_area : ℕ := 2   -- 2x1
def region5_area : ℕ := 7   -- 7x1

-- Define the flower costs
def aster_cost : ℚ := 1
def begonia_cost : ℚ := 2
def canna_cost : ℚ := 2
def dahlia_cost : ℚ := 3
def easter_lily_cost : ℚ := 2.5

-- Define the total garden area
def total_area : ℕ := region1_area + region2_area + region3_area + region4_area + region5_area

-- Theorem statement
theorem min_garden_cost : 
  ∃ (aster_count begonia_count canna_count dahlia_count easter_lily_count : ℕ),
    aster_count + begonia_count + canna_count + dahlia_count + easter_lily_count = total_area ∧
    aster_count * aster_cost + 
    begonia_count * begonia_cost + 
    canna_count * canna_cost + 
    dahlia_count * dahlia_cost + 
    easter_lily_count * easter_lily_cost = 81.5 ∧
    ∀ (a b c d e : ℕ),
      a + b + c + d + e = total_area →
      a * aster_cost + b * begonia_cost + c * canna_cost + d * dahlia_cost + e * easter_lily_cost ≥ 81.5 :=
by sorry

end min_garden_cost_l744_74448


namespace order_of_exponents_l744_74404

theorem order_of_exponents : 5^(1/5) > 0.5^(1/5) ∧ 0.5^(1/5) > 0.5^2 := by
  sorry

end order_of_exponents_l744_74404


namespace expression_evaluation_l744_74482

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  3 * (2 * x^2 * y - x * y^2) - (4 * x^2 * y + x * y^2) = -16 := by
  sorry

end expression_evaluation_l744_74482


namespace original_group_size_l744_74446

theorem original_group_size 
  (total_work : ℝ) 
  (original_days : ℕ) 
  (remaining_days : ℕ) 
  (absent_men : ℕ) :
  let original_work_rate := total_work / original_days
  let remaining_work_rate := total_work / remaining_days
  original_days = 10 ∧ 
  remaining_days = 12 ∧ 
  absent_men = 5 →
  ∃ (original_size : ℕ),
    original_size * original_work_rate = (original_size - absent_men) * remaining_work_rate ∧
    original_size = 25 :=
by sorry

end original_group_size_l744_74446


namespace square_sum_equals_ten_l744_74443

theorem square_sum_equals_ten : 2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10 := by
  sorry

end square_sum_equals_ten_l744_74443
