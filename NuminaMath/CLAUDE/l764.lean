import Mathlib

namespace inverse_variation_problem_l764_76411

/-- Given that x³ and y vary inversely, x and y are always positive, and y = 8 when x = 2,
    prove that x = 2/5 when y = 500. -/
theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ (k : ℝ), ∀ (x y : ℝ), x^3 * y = k) 
  (h4 : 2^3 * 8 = (2 : ℝ)^3 * 8) : 
  (y = 500 → x = 2/5) := by
  sorry

end inverse_variation_problem_l764_76411


namespace first_week_daily_rate_l764_76485

def daily_rate_first_week (x : ℚ) : Prop :=
  ∃ (total_cost : ℚ),
    total_cost = 7 * x + 16 * 11 ∧
    total_cost = 302

theorem first_week_daily_rate :
  ∀ x : ℚ, daily_rate_first_week x → x = 18 :=
by sorry

end first_week_daily_rate_l764_76485


namespace bus_ride_difference_l764_76462

theorem bus_ride_difference (vince_ride : ℝ) (zachary_ride : ℝ)
  (h1 : vince_ride = 0.62)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.12 := by
sorry

end bus_ride_difference_l764_76462


namespace matthew_initial_cakes_l764_76406

/-- The number of friends Matthew has -/
def num_friends : ℕ := 4

/-- The initial number of crackers Matthew has -/
def initial_crackers : ℕ := 10

/-- The number of cakes each person eats -/
def cakes_eaten_per_person : ℕ := 2

/-- The number of crackers given to each friend -/
def crackers_per_friend : ℕ := initial_crackers / num_friends

/-- The initial number of cakes Matthew had -/
def initial_cakes : ℕ := 2 * num_friends * crackers_per_friend

theorem matthew_initial_cakes :
  initial_cakes = 16 :=
sorry

end matthew_initial_cakes_l764_76406


namespace jack_piggy_bank_total_l764_76424

/-- Calculates the final amount in Jack's piggy bank after a given number of weeks -/
def piggy_bank_total (initial_amount : ℝ) (weekly_allowance : ℝ) (savings_rate : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance * savings_rate * weeks)

/-- Proves that Jack will have $83 in his piggy bank after 8 weeks -/
theorem jack_piggy_bank_total :
  piggy_bank_total 43 10 0.5 8 = 83 := by
  sorry

end jack_piggy_bank_total_l764_76424


namespace arithmetic_sequence_length_l764_76445

/-- 
Given an arithmetic sequence with:
- First term a₁ = -5
- Last term aₙ = 40
- Common difference d = 3

Prove that the sequence has 16 terms.
-/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ → ℤ),
  (a 0 = -5) →  -- First term
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference
  (∃ k, a k = 40) →  -- Last term
  (∃ n, n = 16 ∧ a (n - 1) = 40) :=
by sorry

end arithmetic_sequence_length_l764_76445


namespace abes_age_l764_76440

theorem abes_age (present_age : ℕ) : 
  (present_age + (present_age - 7) = 31) → present_age = 19 := by
  sorry

end abes_age_l764_76440


namespace acute_triangle_selection_l764_76486

/-- A point on a circle, with a color attribute -/
structure ColoredPoint where
  point : ℝ × ℝ
  color : Nat

/-- Represents a circle with colored points -/
structure ColoredCircle where
  center : ℝ × ℝ
  radius : ℝ
  points : List ColoredPoint

/-- Checks if three points form an acute or right-angled triangle -/
def isAcuteOrRightTriangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Checks if a ColoredCircle has at least one point of each color (assuming colors are 1, 2, 3) -/
def hasAllColors (circle : ColoredCircle) : Prop := sorry

/-- The main theorem to be proved -/
theorem acute_triangle_selection (circle : ColoredCircle) 
  (h : hasAllColors circle) : 
  ∃ (p1 p2 p3 : ColoredPoint), 
    p1 ∈ circle.points ∧ 
    p2 ∈ circle.points ∧ 
    p3 ∈ circle.points ∧ 
    p1.color ≠ p2.color ∧ 
    p2.color ≠ p3.color ∧ 
    p1.color ≠ p3.color ∧ 
    isAcuteOrRightTriangle p1.point p2.point p3.point := by
  sorry

end acute_triangle_selection_l764_76486


namespace carbonated_water_in_solution2_l764_76414

/-- Represents a solution mixture of lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- Represents the final mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  proportion1 : ℝ
  proportion2 : ℝ
  sum_to_one : proportion1 + proportion2 = 1
  carbonated_water_percent : ℝ

/-- The main theorem to prove -/
theorem carbonated_water_in_solution2 
  (mix : Mixture)
  (h1 : mix.solution1.lemonade = 0.2)
  (h2 : mix.solution2.lemonade = 0.45)
  (h3 : mix.carbonated_water_percent = 0.72)
  (h4 : mix.proportion1 = 0.6799999999999997) :
  mix.solution2.carbonated_water = 0.55 := by
  sorry

#eval 1 - 0.45 -- Expected output: 0.55

end carbonated_water_in_solution2_l764_76414


namespace consecutive_good_numbers_l764_76439

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for a number being "good" (not divisible by the sum of its digits) -/
def is_good (n : ℕ) : Prop := ¬(sum_of_digits n ∣ n)

/-- Main theorem -/
theorem consecutive_good_numbers (n : ℕ) (hn : n > 0) :
  ∃ (start : ℕ), ∀ (i : ℕ), i < n → is_good (start + i) := by sorry

end consecutive_good_numbers_l764_76439


namespace value_of_a_l764_76416

def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

theorem value_of_a : 
  ∃ (a : ℝ), (U a).diff (P a) = {-1} → a = -1 := by
sorry

end value_of_a_l764_76416


namespace exponential_equation_solution_l764_76402

theorem exponential_equation_solution :
  ∃! x : ℝ, 3^(2*x + 2) = (1 : ℝ) / 9 :=
by
  use -2
  sorry

end exponential_equation_solution_l764_76402


namespace opposite_number_theorem_l764_76420

theorem opposite_number_theorem (a : ℝ) : -a = -1 → a + 1 = 2 := by
  sorry

end opposite_number_theorem_l764_76420


namespace function_properties_l764_76460

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem function_properties (f : ℝ → ℝ)
  (h1 : is_odd (λ x ↦ f (x + 2)))
  (h2 : ∀ x₁ x₂, x₁ ∈ Set.Ici 2 → x₂ ∈ Set.Ici 2 → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0) :
  (∀ x y, x < y → f x < f y) ∧
  {x : ℝ | f x < 0} = Set.Iio 2 :=
by sorry

end function_properties_l764_76460


namespace sqrt_product_simplification_l764_76455

theorem sqrt_product_simplification (q : ℝ) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^2) * Real.sqrt (14 * q^3) = 4 * q^3 * Real.sqrt 105 :=
by sorry

end sqrt_product_simplification_l764_76455


namespace simon_kabob_cost_l764_76451

/-- Represents the cost of making kabob sticks -/
def cost_of_kabobs (cubes_per_stick : ℕ) (cubes_per_slab : ℕ) (cost_per_slab : ℕ) (num_sticks : ℕ) : ℕ :=
  let slabs_needed := (num_sticks * cubes_per_stick + cubes_per_slab - 1) / cubes_per_slab
  slabs_needed * cost_per_slab

/-- Proves that the cost for Simon to make 40 kabob sticks is $50 -/
theorem simon_kabob_cost : cost_of_kabobs 4 80 25 40 = 50 := by
  sorry

end simon_kabob_cost_l764_76451


namespace horner_operations_count_l764_76488

/-- Represents a polynomial of degree 6 with a constant term -/
structure Polynomial6 where
  coeffs : Fin 7 → ℝ
  constant_term : coeffs 0 ≠ 0

/-- Counts the number of operations in Horner's method for a polynomial of degree 6 -/
def horner_operations (p : Polynomial6) : ℕ :=
  6 + 6

theorem horner_operations_count (p : Polynomial6) :
  horner_operations p = 12 := by
  sorry

#check horner_operations_count

end horner_operations_count_l764_76488


namespace sign_determination_l764_76423

theorem sign_determination (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a / 5 > 0)
  (h2 : -b / (7*a) > 0)
  (h3 : 11 / (a*b*c) > 0)
  (h4 : -18 / (a*b*c*d) > 0) :
  a > 0 ∧ b < 0 ∧ c < 0 ∧ d < 0 := by
sorry

end sign_determination_l764_76423


namespace sarah_cookies_count_l764_76452

/-- The number of cookies Sarah took -/
def cookies_sarah_took (total_cookies : ℕ) (num_neighbors : ℕ) (cookies_per_neighbor : ℕ) (cookies_left : ℕ) : ℕ :=
  total_cookies - cookies_left - (num_neighbors - 1) * cookies_per_neighbor

theorem sarah_cookies_count :
  cookies_sarah_took 150 15 10 8 = 12 := by
  sorry

end sarah_cookies_count_l764_76452


namespace share_multiple_l764_76453

theorem share_multiple (total : ℚ) (c_share : ℚ) (x : ℚ) : 
  total = 585 →
  c_share = 260 →
  ∃ (a_share b_share : ℚ),
    a_share + b_share + c_share = total ∧
    x * a_share = 6 * b_share ∧
    x * a_share = 3 * c_share →
  x = 4 := by
sorry

end share_multiple_l764_76453


namespace min_value_x_plus_reciprocal_min_value_is_three_l764_76464

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem min_value_is_three : ∃ (x : ℝ), x > 1 ∧ x + 1 / (x - 1) = 3 := by
  sorry

end min_value_x_plus_reciprocal_min_value_is_three_l764_76464


namespace quadratic_always_positive_l764_76434

theorem quadratic_always_positive (k : ℝ) : ∀ x : ℝ, x^2 - (k - 4)*x + k - 7 > 0 := by
  sorry

end quadratic_always_positive_l764_76434


namespace theater_ticket_cost_l764_76458

/-- Proves that the cost of an orchestra seat is $12 given the conditions of the theater ticket sales --/
theorem theater_ticket_cost (balcony_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (balcony_excess : ℕ) :
  balcony_cost = 8 →
  total_tickets = 350 →
  total_revenue = 3320 →
  balcony_excess = 90 →
  ∃ (orchestra_cost : ℕ), 
    orchestra_cost = 12 ∧
    (total_tickets - balcony_excess) / 2 * orchestra_cost + 
    (total_tickets + balcony_excess) / 2 * balcony_cost = total_revenue :=
by
  sorry

#check theater_ticket_cost

end theater_ticket_cost_l764_76458


namespace final_tree_count_l764_76407

/-- 
Given:
- T: The initial number of trees
- P: The percentage of trees cut (as a whole number, e.g., 20 for 20%)
- R: The number of new trees planted for each tree cut

Prove that the final number of trees F is equal to T - (P/100 * T) + (P/100 * T * R)
-/
theorem final_tree_count (T P R : ℕ) (h1 : P ≤ 100) : 
  ∃ F : ℕ, F = T - (P * T / 100) + (P * T * R / 100) :=
sorry

end final_tree_count_l764_76407


namespace perpendicular_planes_line_parallel_l764_76448

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  perpendicular_plane_plane : Plane → Plane → Prop
  perpendicular_line_plane : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop

variable {S : Space3D}

/-- The main theorem -/
theorem perpendicular_planes_line_parallel 
  (α β : S.Plane) (m : S.Line)
  (h1 : S.perpendicular_plane_plane α β)
  (h2 : S.perpendicular_line_plane m β)
  (h3 : ¬ S.line_in_plane m α) :
  S.parallel_line_plane m α :=
sorry

end perpendicular_planes_line_parallel_l764_76448


namespace nina_widget_purchase_l764_76461

/-- Calculates the number of widgets Nina can purchase given her budget and widget price information. -/
def widgets_nina_can_buy (budget : ℕ) (reduced_price_widgets : ℕ) (price_reduction : ℕ) : ℕ :=
  let original_price := (budget + reduced_price_widgets * price_reduction) / reduced_price_widgets
  budget / original_price

/-- Proves that Nina can buy 6 widgets given the problem conditions. -/
theorem nina_widget_purchase :
  widgets_nina_can_buy 48 8 2 = 6 := by
  sorry

#eval widgets_nina_can_buy 48 8 2

end nina_widget_purchase_l764_76461


namespace bowling_ball_difference_l764_76430

theorem bowling_ball_difference :
  ∀ (red green : ℕ),
  red = 30 →
  green > red →
  red + green = 66 →
  green - red = 6 :=
by sorry

end bowling_ball_difference_l764_76430


namespace mark_kangaroos_l764_76413

theorem mark_kangaroos (num_kangaroos num_goats : ℕ) : 
  num_goats = 3 * num_kangaroos →
  2 * num_kangaroos + 4 * num_goats = 322 →
  num_kangaroos = 23 := by
sorry

end mark_kangaroos_l764_76413


namespace simplify_and_evaluate_l764_76450

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  ((2 / (x - 1) - 1 / x) / ((x^2 - 1) / (x^2 - 2*x + 1))) = 1 / x :=
by sorry

end simplify_and_evaluate_l764_76450


namespace sum_of_reciprocals_l764_76470

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end sum_of_reciprocals_l764_76470


namespace wendy_walked_distance_l764_76425

/-- The number of miles Wendy ran -/
def miles_ran : ℝ := 19.83

/-- The difference between miles ran and walked -/
def difference : ℝ := 10.67

/-- The number of miles Wendy walked -/
def miles_walked : ℝ := miles_ran - difference

theorem wendy_walked_distance : miles_walked = 9.16 := by
  sorry

end wendy_walked_distance_l764_76425


namespace five_Y_three_equals_64_l764_76437

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_64 : Y 5 3 = 64 := by
  sorry

end five_Y_three_equals_64_l764_76437


namespace decreasing_function_condition_l764_76435

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x

-- State the theorem
theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (1 < a ∧ a < 2) := by
  sorry

end decreasing_function_condition_l764_76435


namespace hiking_trip_up_rate_l764_76467

/-- Represents the hiking trip parameters -/
structure HikingTrip where
  upRate : ℝ  -- Rate of ascent in miles per day
  downRate : ℝ  -- Rate of descent in miles per day
  upTime : ℝ  -- Time taken for ascent in days
  downTime : ℝ  -- Time taken for descent in days
  downDistance : ℝ  -- Distance of the descent route in miles

/-- The hiking trip satisfies the given conditions -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.upTime = trip.downTime ∧  -- Same time for each route
  trip.downRate = 1.5 * trip.upRate ∧  -- Down rate is 1.5 times up rate
  trip.upTime = 2 ∧  -- 2 days to go up
  trip.downDistance = 9  -- 9 miles down

theorem hiking_trip_up_rate (trip : HikingTrip) 
  (h : validHikingTrip trip) : trip.upRate = 3 := by
  sorry

end hiking_trip_up_rate_l764_76467


namespace exists_special_sequence_l764_76474

/-- A sequence of positive integers satisfying the required properties -/
def SpecialSequence : Type :=
  ℕ → ℕ+

/-- The property that a number has no square factors other than 1 -/
def HasNoSquareFactors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

/-- The main theorem stating the existence of the special sequence -/
theorem exists_special_sequence :
  ∃ (seq : SpecialSequence),
    (∀ i j : ℕ, i < j → seq i < seq j) ∧
    (∀ i j : ℕ, i ≠ j → HasNoSquareFactors ((seq i).val + (seq j).val)) := by
  sorry


end exists_special_sequence_l764_76474


namespace skips_per_second_l764_76497

def minutes_jumped : ℕ := 10
def total_skips : ℕ := 1800

def seconds_jumped : ℕ := minutes_jumped * 60

theorem skips_per_second : total_skips / seconds_jumped = 3 := by
  sorry

end skips_per_second_l764_76497


namespace rabbit_carrots_l764_76403

/-- Represents the number of carrots in each burrow -/
def carrots_per_burrow : ℕ := 2

/-- Represents the number of apples in each tree -/
def apples_per_tree : ℕ := 3

/-- Represents the difference between the number of burrows and trees -/
def burrow_tree_difference : ℕ := 3

theorem rabbit_carrots (burrows trees : ℕ) : 
  burrows = trees + burrow_tree_difference →
  carrots_per_burrow * burrows = apples_per_tree * trees →
  carrots_per_burrow * burrows = 18 := by
  sorry

end rabbit_carrots_l764_76403


namespace reliable_plumbing_hourly_charge_l764_76469

/-- Paul's Plumbing visit charge -/
def paul_visit : ℕ := 55

/-- Paul's Plumbing hourly labor charge -/
def paul_hourly : ℕ := 35

/-- Reliable Plumbing visit charge -/
def reliable_visit : ℕ := 75

/-- Number of labor hours -/
def labor_hours : ℕ := 4

/-- Reliable Plumbing's hourly labor charge -/
def reliable_hourly : ℕ := 30

theorem reliable_plumbing_hourly_charge :
  paul_visit + labor_hours * paul_hourly = reliable_visit + labor_hours * reliable_hourly :=
by sorry

end reliable_plumbing_hourly_charge_l764_76469


namespace rectangle_width_equal_to_square_area_l764_76401

theorem rectangle_width_equal_to_square_area (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) :
  square_side = 8 →
  rect_length = 16 →
  square_side * square_side = rect_length * rect_width →
  rect_width = 4 := by
sorry

end rectangle_width_equal_to_square_area_l764_76401


namespace bobs_small_gate_width_l764_76481

/-- Represents a rectangular garden with gates and fencing -/
structure Garden where
  length : ℝ
  width : ℝ
  large_gate_width : ℝ
  total_fencing : ℝ

/-- Calculates the perimeter of a rectangle -/
def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

/-- Calculates the width of the small gate -/
def small_gate_width (g : Garden) : ℝ :=
  g.total_fencing - rectangle_perimeter g.length g.width + g.large_gate_width

/-- Theorem stating the width of the small gate in Bob's garden -/
theorem bobs_small_gate_width :
  let g : Garden := {
    length := 225,
    width := 125,
    large_gate_width := 10,
    total_fencing := 687
  }
  small_gate_width g = 3 := by
  sorry


end bobs_small_gate_width_l764_76481


namespace factor_polynomial_l764_76444

theorem factor_polynomial (x y : ℝ) : 
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) := by
  sorry

end factor_polynomial_l764_76444


namespace incorrect_value_calculation_l764_76419

/-- Given a set of values with an incorrect mean due to a copying error,
    calculate the incorrect value that was used. -/
theorem incorrect_value_calculation
  (n : ℕ)
  (initial_mean correct_mean : ℚ)
  (correct_value : ℚ)
  (h_n : n = 30)
  (h_initial_mean : initial_mean = 250)
  (h_correct_mean : correct_mean = 251)
  (h_correct_value : correct_value = 165) :
  ∃ (incorrect_value : ℚ),
    incorrect_value = 195 ∧
    n * correct_mean = n * initial_mean - correct_value + incorrect_value :=
by sorry

end incorrect_value_calculation_l764_76419


namespace equal_selection_probability_l764_76489

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a sampling method -/
def selectionProbability (method : SamplingMethod) (N : ℕ) (n : ℕ) : ℝ :=
  sorry

theorem equal_selection_probability (N : ℕ) (n : ℕ) :
  ∀ (m₁ m₂ : SamplingMethod), selectionProbability m₁ N n = selectionProbability m₂ N n :=
  sorry

end equal_selection_probability_l764_76489


namespace jerry_candy_problem_l764_76417

/-- Given a total number of candy pieces, number of bags, and the distribution of chocolate types,
    calculate the number of non-chocolate candy pieces. -/
def non_chocolate_candy (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (kiss_bags : ℕ) : ℕ :=
  total_candy - (heart_bags + kiss_bags) * (total_candy / total_bags)

/-- Theorem stating that given 63 pieces of candy divided into 9 bags,
    with 2 bags of chocolate hearts and 3 bags of chocolate kisses,
    the number of non-chocolate candies is 28. -/
theorem jerry_candy_problem :
  non_chocolate_candy 63 9 2 3 = 28 := by
  sorry

end jerry_candy_problem_l764_76417


namespace rectangle_area_perimeter_sum_l764_76421

theorem rectangle_area_perimeter_sum (w : ℕ) (h : w > 0) : 
  let l := 2 * w
  let A := l * w
  let P := 2 * (l + w)
  A + P ≠ 110 :=
by
  sorry

end rectangle_area_perimeter_sum_l764_76421


namespace f_of_2_equals_negative_2_l764_76405

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_of_2_equals_negative_2 : f 2 = -2 := by
  sorry

end f_of_2_equals_negative_2_l764_76405


namespace calculate_ampersand_composition_l764_76475

-- Define the operations
def ampersand_right (x : ℝ) : ℝ := 10 - x
def ampersand_left (x : ℝ) : ℝ := x - 10

-- State the theorem
theorem calculate_ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end calculate_ampersand_composition_l764_76475


namespace right_triangle_tangent_sum_l764_76410

theorem right_triangle_tangent_sum (α β : Real) (k : Real) : 
  α > 0 → β > 0 → α + β = π / 2 →
  (1 / 2) * Real.cos α * Real.cos β = k →
  Real.tan α + Real.tan β = 2 * k := by
sorry

end right_triangle_tangent_sum_l764_76410


namespace third_place_prize_l764_76404

theorem third_place_prize (total_prize : ℕ) (num_novels : ℕ) (first_prize : ℕ) (second_prize : ℕ) (other_prize : ℕ) :
  total_prize = 800 →
  num_novels = 18 →
  first_prize = 200 →
  second_prize = 150 →
  other_prize = 22 →
  (num_novels - 3) * other_prize + first_prize + second_prize + 120 = total_prize :=
by sorry

end third_place_prize_l764_76404


namespace D_value_l764_76472

/-- The determinant of a matrix with elements |i-j| -/
def D (n : ℕ) : ℚ :=
  let M : Matrix (Fin n) (Fin n) ℚ := λ i j => |i.val - j.val|
  M.det

/-- Theorem stating the value of the determinant D_n -/
theorem D_value (n : ℕ) (h : n > 0) : D n = (-1)^(n-1) * (n-1) * 2^(n-2) := by
  sorry

end D_value_l764_76472


namespace cow_count_is_twenty_l764_76473

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem: In a group where the total number of legs is 40 more than twice
    the number of heads, the number of cows is 20 -/
theorem cow_count_is_twenty (count : AnimalCount) 
    (h : totalLegs count = 2 * totalHeads count + 40) : 
    count.cows = 20 := by
  sorry


end cow_count_is_twenty_l764_76473


namespace total_broadcasting_period_l764_76494

/-- Given a music station that played commercials for a certain duration and maintained a specific ratio of music to commercials, this theorem proves the total broadcasting period. -/
theorem total_broadcasting_period 
  (commercial_duration : ℕ) 
  (music_ratio : ℕ) 
  (commercial_ratio : ℕ) 
  (h1 : commercial_duration = 40)
  (h2 : music_ratio = 9)
  (h3 : commercial_ratio = 5) :
  commercial_duration + (commercial_duration * music_ratio) / commercial_ratio = 112 :=
by sorry

end total_broadcasting_period_l764_76494


namespace fruit_selection_problem_l764_76490

/-- The number of ways to choose n items from k groups with at least m items from each group -/
def choose_with_minimum (n k m : ℕ) : ℕ :=
  (n - k * m + k - 1).choose (k - 1)

/-- The problem statement -/
theorem fruit_selection_problem :
  choose_with_minimum 15 4 2 = 120 := by
  sorry

end fruit_selection_problem_l764_76490


namespace greatest_root_of_g_l764_76433

-- Define the function g(x)
def g (x : ℝ) : ℝ := 18 * x^4 - 20 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = 1 ∧ g r = 0 ∧ ∀ x : ℝ, g x = 0 → x ≤ r :=
by sorry

end greatest_root_of_g_l764_76433


namespace quadratic_root_problem_l764_76477

theorem quadratic_root_problem (m : ℝ) :
  (2 : ℝ)^2 + 2 + m = 0 → ∃ (x : ℝ), x^2 + x + m = 0 ∧ x ≠ 2 ∧ x = -3 :=
by sorry

end quadratic_root_problem_l764_76477


namespace peter_double_harriet_age_l764_76487

def mother_age : ℕ := 60
def harriet_age : ℕ := 13

def peter_age : ℕ := mother_age / 2

def years_until_double (x : ℕ) : Prop :=
  peter_age + x = 2 * (harriet_age + x)

theorem peter_double_harriet_age :
  ∃ x : ℕ, years_until_double x ∧ x = 4 := by
sorry

end peter_double_harriet_age_l764_76487


namespace problem_solution_l764_76447

theorem problem_solution : (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := by
  sorry

end problem_solution_l764_76447


namespace same_shape_proof_l764_76400

/-- A quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two quadratic functions have the same shape if the absolute values of their x² coefficients are equal -/
def same_shape (f g : QuadraticFunction) : Prop :=
  |f.a| = |g.a|

/-- The original function y = 5x² -/
def f : QuadraticFunction :=
  { a := 5, b := 0, c := 0 }

/-- The function y = -5x² + 2 -/
def g : QuadraticFunction :=
  { a := -5, b := 0, c := 2 }

theorem same_shape_proof : same_shape f g := by
  sorry

end same_shape_proof_l764_76400


namespace bob_walking_distance_l764_76427

/-- Proves that Bob walked 4 miles before meeting Yolanda given the problem conditions -/
theorem bob_walking_distance (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (head_start : ℝ) (h1 : total_distance = 10) 
  (h2 : yolanda_rate = 3) (h3 : bob_rate = 4) (h4 : head_start = 1) : 
  ∃ t : ℝ, t > head_start ∧ yolanda_rate * t + bob_rate * (t - head_start) = total_distance ∧ 
  bob_rate * (t - head_start) = 4 :=
by sorry

end bob_walking_distance_l764_76427


namespace plain_lemonade_price_calculation_l764_76459

/-- The price of a glass of plain lemonade -/
def plain_lemonade_price : ℚ := 3 / 4

/-- The number of glasses of plain lemonade sold -/
def plain_lemonade_sold : ℕ := 36

/-- The amount made from strawberry lemonade -/
def strawberry_lemonade_sales : ℕ := 16

/-- The difference between plain and strawberry lemonade sales -/
def sales_difference : ℕ := 11

theorem plain_lemonade_price_calculation :
  plain_lemonade_price * plain_lemonade_sold = 
  (strawberry_lemonade_sales + sales_difference : ℚ) := by sorry

end plain_lemonade_price_calculation_l764_76459


namespace least_clock_equivalent_hour_l764_76442

def is_clock_equivalent (t : ℕ) : Prop :=
  24 ∣ (t^2 - t)

theorem least_clock_equivalent_hour : 
  ∀ t : ℕ, t > 5 → t < 9 → ¬(is_clock_equivalent t) ∧ is_clock_equivalent 9 :=
by sorry

end least_clock_equivalent_hour_l764_76442


namespace book_pages_count_l764_76436

/-- The number of pages Lance read on the first day -/
def pages_day1 : ℕ := 35

/-- The number of pages Lance read on the second day -/
def pages_day2 : ℕ := pages_day1 - 5

/-- The number of pages Lance will read on the third day -/
def pages_day3 : ℕ := 35

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_day1 + pages_day2 + pages_day3

theorem book_pages_count : total_pages = 100 := by
  sorry

end book_pages_count_l764_76436


namespace largest_n_satisfying_inequality_l764_76492

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1 / 4 : ℚ) + (n : ℚ) / 5 < 7 / 4 ↔ n ≤ 7 :=
by sorry

end largest_n_satisfying_inequality_l764_76492


namespace water_depth_difference_l764_76484

theorem water_depth_difference (dean_height : ℝ) (water_depth_factor : ℝ) : 
  dean_height = 9 →
  water_depth_factor = 10 →
  water_depth_factor * dean_height - dean_height = 81 :=
by
  sorry

end water_depth_difference_l764_76484


namespace inverse_g_90_l764_76431

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_g_90 : g 3 = 90 := by sorry

end inverse_g_90_l764_76431


namespace consecutive_integers_sqrt_3_l764_76482

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end consecutive_integers_sqrt_3_l764_76482


namespace inequality_reciprocal_l764_76438

theorem inequality_reciprocal (a b : ℝ) (h : a * b > 0) :
  a > b ↔ 1 / a < 1 / b := by
sorry

end inequality_reciprocal_l764_76438


namespace mixed_fraction_product_l764_76429

theorem mixed_fraction_product (X Y : ℤ) : 
  (5 + 1 / X : ℚ) * (Y + 1 / 2 : ℚ) = 43 →
  5 < 5 + 1 / X →
  5 + 1 / X ≤ 11 / 2 →
  X = 17 ∧ Y = 8 := by
sorry

end mixed_fraction_product_l764_76429


namespace units_digit_of_power_of_three_l764_76418

theorem units_digit_of_power_of_three (n : ℕ) : (3^(4*n + 2) % 10 = 9) := by
  sorry

end units_digit_of_power_of_three_l764_76418


namespace equation_represents_hyperbola_l764_76479

/-- The equation |y-3| = √((x+4)² + 4y²) represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (x y : ℝ), |y - 3| = Real.sqrt ((x + 4)^2 + 4*y^2) →
  ∃ (A B C D E : ℝ), A ≠ 0 ∧ C ≠ 0 ∧ A * C < 0 ∧
    A * y^2 + B * y + C * x^2 + D * x + E = 0 :=
by sorry

end equation_represents_hyperbola_l764_76479


namespace original_price_per_acre_l764_76478

/-- Proves that the original price per acre was $140 --/
theorem original_price_per_acre 
  (total_area : ℕ)
  (sold_area : ℕ)
  (selling_price : ℕ)
  (profit : ℕ)
  (h1 : total_area = 200)
  (h2 : sold_area = total_area / 2)
  (h3 : selling_price = 200)
  (h4 : profit = 6000)
  : (selling_price * sold_area - profit) / sold_area = 140 := by
  sorry

end original_price_per_acre_l764_76478


namespace equal_arcs_equal_chords_l764_76483

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc where
  circle : Circle
  start_angle : ℝ
  end_angle : ℝ

/-- Represents a chord of a circle -/
structure Chord where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Function to calculate the length of an arc -/
def arcLength (arc : Arc) : ℝ := sorry

/-- Function to calculate the length of a chord -/
def chordLength (chord : Chord) : ℝ := sorry

/-- Theorem: In a circle, equal arcs correspond to equal chords -/
theorem equal_arcs_equal_chords (c : Circle) (arc1 arc2 : Arc) (chord1 chord2 : Chord) :
  arc1.circle = c → arc2.circle = c →
  chord1.circle = c → chord2.circle = c →
  arcLength arc1 = arcLength arc2 →
  chord1.endpoint1 = (c.center.1 + c.radius * Real.cos arc1.start_angle,
                      c.center.2 + c.radius * Real.sin arc1.start_angle) →
  chord1.endpoint2 = (c.center.1 + c.radius * Real.cos arc1.end_angle,
                      c.center.2 + c.radius * Real.sin arc1.end_angle) →
  chord2.endpoint1 = (c.center.1 + c.radius * Real.cos arc2.start_angle,
                      c.center.2 + c.radius * Real.sin arc2.start_angle) →
  chord2.endpoint2 = (c.center.1 + c.radius * Real.cos arc2.end_angle,
                      c.center.2 + c.radius * Real.sin arc2.end_angle) →
  chordLength chord1 = chordLength chord2 := by sorry

end equal_arcs_equal_chords_l764_76483


namespace max_children_spell_names_l764_76441

/-- Represents the available letters in the bag -/
def LetterBag : Finset Char := {'A', 'A', 'A', 'A', 'B', 'B', 'D', 'I', 'I', 'M', 'M', 'N', 'N', 'N', 'Y', 'Y'}

/-- Represents the names of the children -/
inductive Child
| Anna
| Vanya
| Dani
| Dima

/-- Returns the set of letters needed to spell a child's name -/
def lettersNeeded (c : Child) : Finset Char :=
  match c with
  | Child.Anna => {'A', 'N', 'N', 'A'}
  | Child.Vanya => {'V', 'A', 'N', 'Y'}
  | Child.Dani => {'D', 'A', 'N', 'Y'}
  | Child.Dima => {'D', 'I', 'M', 'A'}

/-- Theorem stating the maximum number of children who can spell their names -/
theorem max_children_spell_names :
  ∃ (S : Finset Child), (∀ c ∈ S, lettersNeeded c ⊆ LetterBag) ∧ 
                        (∀ T : Finset Child, (∀ c ∈ T, lettersNeeded c ⊆ LetterBag) → T.card ≤ S.card) ∧
                        S.card = 3 := by
  sorry

end max_children_spell_names_l764_76441


namespace greater_number_problem_l764_76408

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : 
  max x y = 35 := by
  sorry

end greater_number_problem_l764_76408


namespace concert_attendance_l764_76463

theorem concert_attendance (num_buses : ℕ) (students_per_bus : ℕ) (students_in_minivan : ℕ) :
  num_buses = 12 →
  students_per_bus = 38 →
  students_in_minivan = 5 →
  num_buses * students_per_bus + students_in_minivan = 461 := by
  sorry

end concert_attendance_l764_76463


namespace double_negation_2023_l764_76454

theorem double_negation_2023 : -(-2023) = 2023 := by
  sorry

end double_negation_2023_l764_76454


namespace binary_10101_is_21_l764_76422

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_is_21_l764_76422


namespace expression_evaluation_l764_76491

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (2*x + y)^2 + (x + y)*(x - y) = -3 := by
  sorry

end expression_evaluation_l764_76491


namespace mindmaster_secret_codes_l764_76466

/-- The number of different colors available for pegs -/
def num_colors : ℕ := 8

/-- The number of slots in the code -/
def num_slots : ℕ := 4

/-- The total number of options for each slot (colors + empty) -/
def options_per_slot : ℕ := num_colors + 1

/-- The number of possible secret codes in the Mindmaster variation -/
theorem mindmaster_secret_codes :
  (options_per_slot ^ num_slots) - 1 = 6560 := by sorry

end mindmaster_secret_codes_l764_76466


namespace merchant_pricing_strategy_l764_76499

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (purchase_price_ratio : ℝ) 
  (discount_ratio : ℝ) 
  (profit_ratio : ℝ) 
  (marked_price_ratio : ℝ) 
  (h1 : purchase_price_ratio = 0.7) 
  (h2 : discount_ratio = 0.25) 
  (h3 : profit_ratio = 0.3) 
  (h4 : marked_price_ratio * (1 - discount_ratio) * list_price - 
        purchase_price_ratio * list_price = 
        profit_ratio * marked_price_ratio * (1 - discount_ratio) * list_price) :
  marked_price_ratio = 1.33 := by
  sorry

#check merchant_pricing_strategy

end merchant_pricing_strategy_l764_76499


namespace grandfather_age_proof_l764_76465

/-- The age of Xiaoming's grandfather -/
def grandfather_age : ℕ := 79

/-- The result after processing the grandfather's age -/
def processed_age (age : ℕ) : ℕ :=
  ((age - 15) / 4 - 6) * 10

theorem grandfather_age_proof :
  processed_age grandfather_age = 100 :=
by sorry

end grandfather_age_proof_l764_76465


namespace find_divisor_l764_76426

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 997) (h2 : quotient = 43) (h3 : remainder = 8) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 23 :=
by sorry

end find_divisor_l764_76426


namespace point_product_theorem_l764_76432

theorem point_product_theorem : 
  ∀ y₁ y₂ : ℝ, 
    ((-2 - 4)^2 + (y₁ - (-1))^2 = 8^2) → 
    ((-2 - 4)^2 + (y₂ - (-1))^2 = 8^2) → 
    y₁ ≠ y₂ →
    y₁ * y₂ = -27 := by
  sorry

end point_product_theorem_l764_76432


namespace integer_pairs_satisfying_conditions_l764_76495

theorem integer_pairs_satisfying_conditions :
  ∀ m n : ℤ, 
    m^2 = n^5 + n^4 + 1 ∧ 
    (m - 7*n) ∣ (m - 4*n) → 
    ((m = -1 ∧ n = 0) ∨ (m = 1 ∧ n = 0)) := by
  sorry

end integer_pairs_satisfying_conditions_l764_76495


namespace fruit_store_inventory_l764_76457

/-- Represents the fruit store inventory and gift basket composition. -/
structure FruitStore where
  cantaloupes : ℕ
  dragonFruits : ℕ
  kiwis : ℕ
  basketCantaloupes : ℕ
  basketDragonFruits : ℕ
  basketKiwis : ℕ

/-- Theorem stating the original number of dragon fruits and remaining kiwis. -/
theorem fruit_store_inventory (store : FruitStore)
  (h1 : store.basketCantaloupes = 2)
  (h2 : store.basketDragonFruits = 4)
  (h3 : store.basketKiwis = 10)
  (h4 : store.dragonFruits = 3 * store.cantaloupes + 10)
  (h5 : store.kiwis = 2 * store.dragonFruits)
  (h6 : store.dragonFruits - store.basketDragonFruits * store.cantaloupes = 130) :
  store.dragonFruits = 370 ∧ 
  store.kiwis - store.basketKiwis * store.cantaloupes = 140 := by
  sorry


end fruit_store_inventory_l764_76457


namespace books_on_third_shelf_l764_76480

/-- Represents the number of books on each shelf of a bookcase -/
structure Bookcase where
  shelf1 : ℕ
  shelf2 : ℕ
  shelf3 : ℕ

/-- Defines the properties of the bookcase in the problem -/
def ProblemBookcase (b : Bookcase) : Prop :=
  b.shelf1 + b.shelf2 + b.shelf3 = 275 ∧
  b.shelf3 = 3 * b.shelf2 + 8 ∧
  b.shelf1 = 2 * b.shelf2 - 3

theorem books_on_third_shelf :
  ∀ b : Bookcase, ProblemBookcase b → b.shelf3 = 188 :=
by
  sorry


end books_on_third_shelf_l764_76480


namespace jane_current_age_jane_age_is_40_l764_76428

theorem jane_current_age : ℕ → Prop :=
  fun jane_age =>
    ∃ (babysitting_start_age babysitting_end_age oldest_babysat_age : ℕ),
      babysitting_start_age = 18 ∧
      babysitting_end_age = jane_age - 10 ∧
      oldest_babysat_age = 25 ∧
      (∀ child_age : ℕ, child_age ≤ oldest_babysat_age - 10 → 2 * child_age ≤ babysitting_end_age) ∧
      jane_age = 40

theorem jane_age_is_40 : jane_current_age 40 := by
  sorry

end jane_current_age_jane_age_is_40_l764_76428


namespace arccos_one_over_sqrt_two_l764_76449

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l764_76449


namespace strawberries_per_basket_is_15_l764_76468

/-- The number of strawberries in each basket picked by Kimberly's brother -/
def strawberries_per_basket (kimberly_amount : ℕ) (brother_baskets : ℕ) (parents_amount : ℕ) (total_amount : ℕ) : ℕ :=
  (total_amount / 4) / brother_baskets

/-- Theorem stating the number of strawberries in each basket picked by Kimberly's brother -/
theorem strawberries_per_basket_is_15 
  (kimberly_amount : ℕ) 
  (brother_baskets : ℕ) 
  (parents_amount : ℕ) 
  (total_amount : ℕ) 
  (h1 : kimberly_amount = 8 * (brother_baskets * strawberries_per_basket kimberly_amount brother_baskets parents_amount total_amount))
  (h2 : parents_amount = kimberly_amount - 93)
  (h3 : brother_baskets = 3)
  (h4 : total_amount = 4 * 168)
  : strawberries_per_basket kimberly_amount brother_baskets parents_amount total_amount = 15 :=
sorry


end strawberries_per_basket_is_15_l764_76468


namespace sum_of_divisors_30_l764_76496

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 := by
  sorry

end sum_of_divisors_30_l764_76496


namespace bmw_sales_count_l764_76493

def total_cars : ℕ := 300

def audi_percent : ℚ := 10 / 100
def toyota_percent : ℚ := 15 / 100
def acura_percent : ℚ := 20 / 100
def honda_percent : ℚ := 18 / 100

def other_brands_percent : ℚ := audi_percent + toyota_percent + acura_percent + honda_percent

def bmw_percent : ℚ := 1 - other_brands_percent

theorem bmw_sales_count : ⌊(bmw_percent * total_cars : ℚ)⌋ = 111 := by
  sorry

end bmw_sales_count_l764_76493


namespace vector_magnitude_problem_l764_76476

/-- Given two vectors a and b in ℝ², prove that if |a| = 1, |b| = 2, and a + b = (2√2, 1), then |3a + b| = 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  norm a = 1 →
  norm b = 2 →
  a + b = (2 * Real.sqrt 2, 1) →
  norm (3 • a + b) = 5 := by
  sorry

end vector_magnitude_problem_l764_76476


namespace ad_agency_client_distribution_l764_76456

/-- Given an advertising agency with 180 clients, where:
    - 115 use television
    - 110 use radio
    - 130 use magazines
    - 85 use television and magazines
    - 75 use television and radio
    - 80 use all three
    This theorem proves that 95 clients use radio and magazines. -/
theorem ad_agency_client_distribution (total : ℕ) (T R M TM TR TRM : ℕ) 
  (h_total : total = 180)
  (h_T : T = 115)
  (h_R : R = 110)
  (h_M : M = 130)
  (h_TM : TM = 85)
  (h_TR : TR = 75)
  (h_TRM : TRM = 80)
  : total = T + R + M - TR - TM - (T + R + M - TR - TM - total + TRM) + TRM := by
  sorry

end ad_agency_client_distribution_l764_76456


namespace unique_c_for_unique_solution_l764_76471

/-- The quadratic equation in x with parameter b -/
def quadratic (b : ℝ) (c : ℝ) (x : ℝ) : Prop :=
  x^2 + (b^2 + 3*b + 1/b)*x + c = 0

/-- The statement to be proved -/
theorem unique_c_for_unique_solution :
  ∃! c : ℝ, c ≠ 0 ∧
    ∃! b : ℝ, b > 0 ∧
      (∃! x : ℝ, quadratic b c x) ∧
      c = -1/2 := by
  sorry

end unique_c_for_unique_solution_l764_76471


namespace f_properties_l764_76443

def f (x : ℝ) := |x - 2|

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
  sorry

end f_properties_l764_76443


namespace inverse_proportion_k_value_l764_76498

/-- Given an inverse proportion function f(x) = k/x where k ≠ 0 and 1 ≤ x ≤ 3,
    if the difference between the maximum and minimum values of f(x) is 4,
    then k = ±6 -/
theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ k / x
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ f 1 ∧ f x ≥ f 3) →
  f 1 - f 3 = 4 →
  k = 6 ∨ k = -6 := by
sorry

end inverse_proportion_k_value_l764_76498


namespace min_candies_removed_correct_l764_76412

/-- Represents the number of candies of each flavor -/
structure CandyCounts where
  chocolate : Nat
  mint : Nat
  butterscotch : Nat

/-- The initial candy counts in the bag -/
def initialCandies : CandyCounts :=
  { chocolate := 4, mint := 6, butterscotch := 10 }

/-- The total number of candies in the bag -/
def totalCandies : Nat := 20

/-- The minimum number of candies that must be removed to ensure
    at least two of each flavor have been eaten -/
def minCandiesRemoved : Nat := 18

theorem min_candies_removed_correct :
  minCandiesRemoved = totalCandies - (initialCandies.chocolate - 2) - (initialCandies.mint - 2) - (initialCandies.butterscotch - 2) :=
by sorry

end min_candies_removed_correct_l764_76412


namespace arithmetic_sequence_common_difference_l764_76409

/-- An arithmetic sequence with 12 terms -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of odd-numbered terms is 10 -/
def SumOddTerms (a : ℕ → ℚ) : Prop :=
  (a 1) + (a 3) + (a 5) + (a 7) + (a 9) + (a 11) = 10

/-- The sum of even-numbered terms is 22 -/
def SumEvenTerms (a : ℕ → ℚ) : Prop :=
  (a 2) + (a 4) + (a 6) + (a 8) + (a 10) + (a 12) = 22

/-- The common difference of the arithmetic sequence is 2 -/
def CommonDifferenceIsTwo (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h1 : ArithmeticSequence a)
  (h2 : SumOddTerms a)
  (h3 : SumEvenTerms a) :
  CommonDifferenceIsTwo a :=
sorry

end arithmetic_sequence_common_difference_l764_76409


namespace unique_triple_l764_76415

/-- A function that checks if a number is divisible by any prime less than 2014 -/
def not_divisible_by_small_primes (n : ℕ) : Prop :=
  ∀ p, p < 2014 → Nat.Prime p → ¬(p ∣ n)

/-- The main theorem statement -/
theorem unique_triple : 
  ∃! (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ n : ℕ, n > 0 → not_divisible_by_small_primes n → 
      (n + c) ∣ (a^n + b^n + n)) ∧
    a = 1 ∧ b = 1 ∧ c = 2 :=
sorry

end unique_triple_l764_76415


namespace store_profit_analysis_l764_76446

/-- Represents the relationship between sales volume and selling price -/
def sales_volume (x : ℝ) : ℝ := -x + 120

/-- Represents the profit function -/
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - 60)

/-- The cost price per item -/
def cost_price : ℝ := 60

/-- The maximum allowed profit percentage -/
def max_profit_percentage : ℝ := 0.45

theorem store_profit_analysis 
  (h1 : ∀ x, x ≥ cost_price)  -- Selling price not lower than cost price
  (h2 : ∀ x, profit x ≤ max_profit_percentage * cost_price * (x - cost_price))  -- Profit not exceeding 45%
  : 
  (∃ max_profit_price : ℝ, 
    max_profit_price = 87 ∧ 
    profit max_profit_price = 891 ∧ 
    ∀ x, profit x ≤ profit max_profit_price) ∧ 
  (∀ x, profit x ≥ 500 ↔ 70 ≤ x ∧ x ≤ 110) := by
  sorry


end store_profit_analysis_l764_76446
