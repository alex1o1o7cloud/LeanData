import Mathlib

namespace NUMINAMATH_CALUDE_machine_output_l4054_405431

/-- The number of shirts an industrial machine can make in a minute. -/
def shirts_per_minute : ℕ := sorry

/-- The number of minutes the machine worked today. -/
def minutes_worked_today : ℕ := 12

/-- The total number of shirts made today. -/
def total_shirts_today : ℕ := 72

/-- Theorem stating that the machine can make 6 shirts per minute. -/
theorem machine_output : shirts_per_minute = 6 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_l4054_405431


namespace NUMINAMATH_CALUDE_expression_evaluation_l4054_405412

theorem expression_evaluation : (2^10 * 3^3) / (6 * 2^5) = 144 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4054_405412


namespace NUMINAMATH_CALUDE_simplify_expression_l4054_405447

theorem simplify_expression (a : ℝ) (h : a < 2) : 
  Real.sqrt ((a - 2)^2) + a - 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l4054_405447


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l4054_405419

/-- Systematic sampling function that returns the nth sample given a starting point and interval -/
def systematic_sample (start : Nat) (interval : Nat) (n : Nat) : Nat :=
  start + (n - 1) * interval

theorem systematic_sampling_fourth_student 
  (total_students : Nat) 
  (sample_size : Nat) 
  (sample1 sample2 sample3 : Nat) :
  total_students = 36 →
  sample_size = 4 →
  sample1 = 6 →
  sample2 = 24 →
  sample3 = 33 →
  ∃ (start interval : Nat),
    systematic_sample start interval 1 = sample1 ∧
    systematic_sample start interval 2 = sample2 ∧
    systematic_sample start interval 3 = sample3 ∧
    systematic_sample start interval 4 = 15 :=
by
  sorry

#check systematic_sampling_fourth_student

end NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l4054_405419


namespace NUMINAMATH_CALUDE_anya_erasers_difference_l4054_405417

theorem anya_erasers_difference (andrea_erasers : ℕ) (anya_ratio : ℚ) : 
  andrea_erasers = 6 → 
  anya_ratio = 4.5 → 
  (anya_ratio * andrea_erasers : ℚ) - andrea_erasers = 21 := by
sorry

end NUMINAMATH_CALUDE_anya_erasers_difference_l4054_405417


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l4054_405425

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 336) → (a + b + c = 21) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l4054_405425


namespace NUMINAMATH_CALUDE_map_scale_l4054_405470

/-- Given a map where 15 centimeters represents 90 kilometers,
    prove that 20 centimeters represents 120 kilometers. -/
theorem map_scale (map_cm : ℝ) (map_km : ℝ) (length_cm : ℝ) :
  map_cm = 15 ∧ map_km = 90 ∧ length_cm = 20 →
  (length_cm / map_cm) * map_km = 120 := by
sorry

end NUMINAMATH_CALUDE_map_scale_l4054_405470


namespace NUMINAMATH_CALUDE_chess_tournament_players_l4054_405420

/-- A chess tournament with specific point distribution rules -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest-scoring group
  total_players : ℕ := n + 15
  
  -- Each player plays exactly one game against every other player
  games_played : ℕ := total_players * (total_players - 1) / 2
  
  -- Points from games between non-lowest scoring players
  points_upper : ℕ := n * (n - 1) / 2
  
  -- Points from games within lowest scoring group
  points_lower : ℕ := 105
  
  -- Conditions on point distribution
  point_distribution : Prop := 
    2 * points_upper + 2 * points_lower = games_played

/-- The theorem stating that the total number of players is 36 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l4054_405420


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l4054_405460

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a*b + a*c + b*c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1008 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l4054_405460


namespace NUMINAMATH_CALUDE_two_p_plus_q_l4054_405432

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = (19 / 7) * q := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l4054_405432


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l4054_405403

theorem solution_to_system_of_equations :
  let solutions : List (ℝ × ℝ) := [
    (Real.sqrt 5, Real.sqrt 6), (Real.sqrt 5, -Real.sqrt 6),
    (-Real.sqrt 5, Real.sqrt 6), (-Real.sqrt 5, -Real.sqrt 6),
    (Real.sqrt 6, Real.sqrt 5), (Real.sqrt 6, -Real.sqrt 5),
    (-Real.sqrt 6, Real.sqrt 5), (-Real.sqrt 6, -Real.sqrt 5)
  ]
  ∀ (x y : ℝ),
    (3 * x^2 + 3 * y^2 - x^2 * y^2 = 3 ∧
     x^4 + y^4 - x^2 * y^2 = 31) ↔
    (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l4054_405403


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l4054_405430

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflectOverYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Sum of coordinates of two points -/
def sumCoordinates (p1 p2 : Point2D) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_sum_coordinates (a : ℝ) :
  let C : Point2D := { x := a, y := 8 }
  let D : Point2D := reflectOverYAxis C
  sumCoordinates C D = 16 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l4054_405430


namespace NUMINAMATH_CALUDE_composite_power_plus_four_l4054_405490

theorem composite_power_plus_four (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_power_plus_four_l4054_405490


namespace NUMINAMATH_CALUDE_jons_laundry_capacity_l4054_405436

/-- Given information about Jon's laundry and machine capacity -/
structure LaundryInfo where
  shirts_per_pound : ℕ  -- Number of shirts that weigh 1 pound
  pants_per_pound : ℕ   -- Number of pairs of pants that weigh 1 pound
  total_shirts : ℕ      -- Total number of shirts to wash
  total_pants : ℕ       -- Total number of pants to wash
  loads : ℕ             -- Number of loads Jon has to do

/-- Calculate the machine capacity given laundry information -/
def machine_capacity (info : LaundryInfo) : ℚ :=
  let shirt_weight := info.total_shirts / info.shirts_per_pound
  let pants_weight := info.total_pants / info.pants_per_pound
  let total_weight := shirt_weight + pants_weight
  total_weight / info.loads

/-- Theorem stating Jon's laundry machine capacity -/
theorem jons_laundry_capacity :
  let info : LaundryInfo := {
    shirts_per_pound := 4,
    pants_per_pound := 2,
    total_shirts := 20,
    total_pants := 20,
    loads := 3
  }
  machine_capacity info = 5 := by sorry

end NUMINAMATH_CALUDE_jons_laundry_capacity_l4054_405436


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_circumcentre_l4054_405488

/-- A point in the Euclidean plane -/
structure Point : Type :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line : Type :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of a cyclic quadrilateral -/
def is_cyclic_quadrilateral (A B X C O : Point) : Prop := sorry

/-- Definition of a point lying on a line -/
def point_on_line (P : Point) (L : Line) : Prop := sorry

/-- Definition of equality of distances -/
def distance_eq (A B C D : Point) : Prop := sorry

/-- Definition of a circumcentre of a triangle -/
def is_circumcentre (O : Point) (A B C : Point) : Prop := sorry

/-- Definition of a perpendicular bisector of a line segment -/
def perpendicular_bisector (L : Line) (A B : Point) : Prop := sorry

/-- Definition of a point lying on a perpendicular bisector -/
def point_on_perp_bisector (P : Point) (L : Line) (A B : Point) : Prop := sorry

theorem cyclic_quadrilateral_circumcentre 
  (A B X C O D E : Point) (BX CX : Line) :
  is_cyclic_quadrilateral A B X C O →
  point_on_line D BX →
  point_on_line E CX →
  distance_eq A D B D →
  distance_eq A E C E →
  ∃ (O₁ : Point), is_circumcentre O₁ D E X ∧
    ∃ (L : Line), perpendicular_bisector L O A ∧ point_on_perp_bisector O₁ L O A :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_circumcentre_l4054_405488


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l4054_405444

-- Define the type for a single die roll
def DieRoll : Type := Fin 6

-- Define the sample space for two die rolls
def SampleSpace : Type := DieRoll × DieRoll

-- Define the probability measure
noncomputable def prob : Set SampleSpace → ℝ := sorry

-- Define the event "sum is 5"
def sum_is_5 (roll : SampleSpace) : Prop :=
  roll.1.val + roll.2.val + 2 = 5

-- Define the event "at least one roll is odd"
def at_least_one_odd (roll : SampleSpace) : Prop :=
  roll.1.val % 2 = 0 ∨ roll.2.val % 2 = 0

-- State the theorem
theorem die_roll_probabilities :
  (prob {roll : SampleSpace | sum_is_5 roll} = 1/9) ∧
  (prob {roll : SampleSpace | at_least_one_odd roll} = 3/4) := by sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l4054_405444


namespace NUMINAMATH_CALUDE_log_difference_l4054_405493

theorem log_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) :
  b - d = 93 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_l4054_405493


namespace NUMINAMATH_CALUDE_solutions_count_l4054_405438

theorem solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + p.2 = 100) 
    (Finset.product (Finset.range 101) (Finset.range 101))).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_l4054_405438


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l4054_405466

/-- Given a bus that stops for half an hour every hour and has an average speed of 6 km/hr including stoppages, 
    its speed excluding stoppages is 12 km/hr. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 0.5) -- 30 minutes = 0.5 hours
  (h2 : avg_speed_with_stops = 6) :
  avg_speed_with_stops / (1 - stop_time) = 12 := by
  sorry

#check bus_speed_excluding_stoppages

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l4054_405466


namespace NUMINAMATH_CALUDE_bike_speed_l4054_405418

/-- Proves that a bike moving at constant speed, covering 32 meters in 8 seconds, has a speed of 4 meters per second -/
theorem bike_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 32 → time = 8 → speed = distance / time → speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_bike_speed_l4054_405418


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l4054_405410

theorem smallest_number_divisible (n : ℕ) : n = 257 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) % 8 = 0 ∧ (m + 7) % 11 = 0 ∧ (m + 7) % 24 = 0)) ∧
  (n + 7) % 8 = 0 ∧ (n + 7) % 11 = 0 ∧ (n + 7) % 24 = 0 :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l4054_405410


namespace NUMINAMATH_CALUDE_intersection_theorem_l4054_405411

def P : Set ℝ := {x | (x - 1)^2 > 16}
def Q (a : ℝ) : Set ℝ := {x | x^2 + (a - 8) * x - 8 * a ≤ 0}

theorem intersection_theorem (a : ℝ) :
  (∃ a, a = 3 → P ∩ Q a = {x | 5 < x ∧ x ≤ 8}) ∧
  (P ∩ Q a = {x | 5 < x ∧ x ≤ 8} ↔ a ∈ Set.Icc (-5) 3) ∧
  (∀ a,
    (a > 3 → P ∩ Q a = {x | -a ≤ x ∧ x < -3 ∨ 5 < x ∧ x ≤ 8}) ∧
    (-5 ≤ a ∧ a ≤ 3 → P ∩ Q a = {x | 5 < x ∧ x ≤ 8}) ∧
    (-8 ≤ a ∧ a < -5 → P ∩ Q a = {x | -a ≤ x ∧ x ≤ 8}) ∧
    (a < -8 → P ∩ Q a = {x | 8 ≤ x ∧ x ≤ -a})) :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_l4054_405411


namespace NUMINAMATH_CALUDE_soccer_field_kids_l4054_405452

/-- Given an initial number of kids on a soccer field and the number of friends each kid invites,
    calculate the total number of kids on the field after invitations. -/
def total_kids_after_invitations (initial_kids : ℕ) (friends_per_kid : ℕ) : ℕ :=
  initial_kids + initial_kids * friends_per_kid

/-- Theorem: If there are initially 14 kids on a soccer field and each kid invites 3 friends,
    then the total number of kids on the field after invitations is 56. -/
theorem soccer_field_kids : total_kids_after_invitations 14 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l4054_405452


namespace NUMINAMATH_CALUDE_k_range_l4054_405423

open Real

/-- The function f(x) = (ln x)/x - kx is increasing on (0, +∞) -/
def f_increasing (k : ℝ) : Prop :=
  ∀ x, x > 0 → Monotone (λ x => (log x) / x - k * x)

/-- The theorem to be proved -/
theorem k_range (k : ℝ) : f_increasing k → k ≤ -1 / (2 * Real.exp 3) := by
  sorry

end NUMINAMATH_CALUDE_k_range_l4054_405423


namespace NUMINAMATH_CALUDE_chemistry_students_l4054_405499

def basketball_team : ℕ := 18
def math_students : ℕ := 10
def physics_students : ℕ := 6
def math_and_physics : ℕ := 3
def all_three : ℕ := 2

theorem chemistry_students : ℕ := by
  -- The number of students studying chemistry is 7
  have h : basketball_team = math_students + physics_students - math_and_physics + (basketball_team - (math_students + physics_students - math_and_physics)) := by sorry
  -- Proof goes here
  sorry

#check chemistry_students -- Should evaluate to 7

end NUMINAMATH_CALUDE_chemistry_students_l4054_405499


namespace NUMINAMATH_CALUDE_value_of_2a_plus_b_l4054_405477

-- Define the functions f, g, and h
def f (a b : ℝ) (x : ℝ) : ℝ := a * x - b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_plus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_value_of_2a_plus_b_l4054_405477


namespace NUMINAMATH_CALUDE_tom_age_is_nine_l4054_405479

/-- Tom's age is 1 year less than twice his sister's age and their ages sum to 14 years. -/
def age_problem (tom_age sister_age : ℕ) : Prop :=
  tom_age = 2 * sister_age - 1 ∧ tom_age + sister_age = 14

/-- Tom's age is 9 years given the conditions. -/
theorem tom_age_is_nine :
  ∃ (sister_age : ℕ), age_problem 9 sister_age :=
by sorry

end NUMINAMATH_CALUDE_tom_age_is_nine_l4054_405479


namespace NUMINAMATH_CALUDE_dolphins_score_l4054_405484

theorem dolphins_score (total_score winning_margin : ℕ) : 
  total_score = 48 → winning_margin = 20 → 
  ∃ (sharks_score dolphins_score : ℕ), 
    sharks_score + dolphins_score = total_score ∧ 
    sharks_score = dolphins_score + winning_margin ∧
    dolphins_score = 14 := by
  sorry

end NUMINAMATH_CALUDE_dolphins_score_l4054_405484


namespace NUMINAMATH_CALUDE_fish_catch_problem_l4054_405433

theorem fish_catch_problem (total_fish : ℕ) 
  (first_fisherman_carp_ratio : ℚ) (second_fisherman_perch_ratio : ℚ) :
  total_fish = 70 ∧ 
  first_fisherman_carp_ratio = 5 / 9 ∧ 
  second_fisherman_perch_ratio = 7 / 17 →
  ∃ (first_catch second_catch : ℕ),
    first_catch + second_catch = total_fish ∧
    first_catch * first_fisherman_carp_ratio = 
      second_catch * second_fisherman_perch_ratio ∧
    first_catch = 36 ∧ 
    second_catch = 34 := by
  sorry

#check fish_catch_problem

end NUMINAMATH_CALUDE_fish_catch_problem_l4054_405433


namespace NUMINAMATH_CALUDE_investment_problem_l4054_405462

/-- Proves that given the investment conditions, the amount invested at Speedy Growth Bank is $300 --/
theorem investment_problem (total_investment : ℝ) (speedy_rate : ℝ) (safe_rate : ℝ) (final_amount : ℝ)
  (h1 : total_investment = 1500)
  (h2 : speedy_rate = 0.04)
  (h3 : safe_rate = 0.06)
  (h4 : final_amount = 1584)
  (h5 : ∀ x : ℝ, x * (1 + speedy_rate) + (total_investment - x) * (1 + safe_rate) = final_amount) :
  ∃ x : ℝ, x = 300 ∧ x * (1 + speedy_rate) + (total_investment - x) * (1 + safe_rate) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l4054_405462


namespace NUMINAMATH_CALUDE_church_seating_capacity_l4054_405422

theorem church_seating_capacity (chairs_per_row : ℕ) (num_rows : ℕ) (total_people : ℕ) :
  chairs_per_row = 6 →
  num_rows = 20 →
  total_people = 600 →
  total_people / (chairs_per_row * num_rows) = 5 :=
by sorry

end NUMINAMATH_CALUDE_church_seating_capacity_l4054_405422


namespace NUMINAMATH_CALUDE_ten_percent_of_n_l4054_405429

theorem ten_percent_of_n (n f : ℝ) (h : n - (1/4 * 2) - (1/3 * 3) - f * n = 27) : 
  (0.1 : ℝ) * n = (0.1 : ℝ) * (28.5 / (1 - f)) := by
sorry

end NUMINAMATH_CALUDE_ten_percent_of_n_l4054_405429


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l4054_405424

/-- 
Given a furniture shop where the owner charges 25% more than the cost price,
this theorem proves that if a customer pays Rs. 1000 for an item, 
then the cost price of that item is Rs. 800.
-/
theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : markup_percentage = 25)
  (h2 : selling_price = 1000) :
  let cost_price := selling_price / (1 + markup_percentage / 100)
  cost_price = 800 := by
  sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l4054_405424


namespace NUMINAMATH_CALUDE_pupils_in_singing_only_l4054_405413

/-- Given a class with pupils in debate and singing activities, calculate the number of pupils in singing only. -/
theorem pupils_in_singing_only
  (total : ℕ)
  (debate_only : ℕ)
  (both : ℕ)
  (h_total : total = 55)
  (h_debate_only : debate_only = 10)
  (h_both : both = 17) :
  total - debate_only - both = 45 :=
by sorry

end NUMINAMATH_CALUDE_pupils_in_singing_only_l4054_405413


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l4054_405449

def total_missiles : ℕ := 60
def selected_missiles : ℕ := 6

def systematic_sample (total : ℕ) (select : ℕ) : List ℕ :=
  let interval := total / select
  List.range select |>.map (fun i => i * interval + interval / 2 + 1)

theorem correct_systematic_sample :
  systematic_sample total_missiles selected_missiles = [3, 13, 23, 33, 43, 53] :=
sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l4054_405449


namespace NUMINAMATH_CALUDE_cuboid_volume_calculation_l4054_405492

def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

theorem cuboid_volume_calculation :
  let length : ℝ := 6
  let width : ℝ := 5
  let height : ℝ := 6
  cuboid_volume length width height = 180 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_calculation_l4054_405492


namespace NUMINAMATH_CALUDE_min_abs_z_complex_l4054_405409

theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 3*I) + Complex.abs (z - 4) = 5) :
  ∃ (min_abs : ℝ), min_abs = 12/5 ∧ ∀ w : ℂ, Complex.abs (w - 3*I) + Complex.abs (w - 4) = 5 → Complex.abs w ≥ min_abs :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_complex_l4054_405409


namespace NUMINAMATH_CALUDE_guitar_payment_plan_l4054_405458

theorem guitar_payment_plan (total_with_interest : ℝ) (num_months : ℕ) (interest_rate : ℝ) :
  total_with_interest = 1320 →
  num_months = 12 →
  interest_rate = 0.1 →
  ∃ (monthly_payment : ℝ),
    monthly_payment * num_months * (1 + interest_rate) = total_with_interest ∧
    monthly_payment = 100 := by
  sorry

end NUMINAMATH_CALUDE_guitar_payment_plan_l4054_405458


namespace NUMINAMATH_CALUDE_square_hexagon_cannot_cover_l4054_405475

-- Define the shapes
inductive Shape
  | Triangle
  | Square
  | Hexagon
  | Octagon

-- Define the internal angle of each shape
def internal_angle (s : Shape) : ℝ :=
  match s with
  | Shape.Triangle => 60
  | Shape.Square => 90
  | Shape.Hexagon => 120
  | Shape.Octagon => 135

-- Define a function to check if two shapes can cover a surface
def can_cover_surface (s1 s2 : Shape) : Prop :=
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * internal_angle s1 + n * internal_angle s2 = 360

-- Theorem statement
theorem square_hexagon_cannot_cover :
  ∀ (s1 s2 : Shape), s1 ≠ s2 →
    (s1 = Shape.Square ∧ s2 = Shape.Hexagon) ∨ (s1 = Shape.Hexagon ∧ s2 = Shape.Square) →
    ¬(can_cover_surface s1 s2) :=
  sorry

#check square_hexagon_cannot_cover

end NUMINAMATH_CALUDE_square_hexagon_cannot_cover_l4054_405475


namespace NUMINAMATH_CALUDE_nine_women_eighteen_tea_l4054_405445

/-- The time (in minutes) it takes for a given number of women to drink a given amount of tea,
    given that 1.5 women drink 1.5 tea in 1.5 minutes. -/
def drinking_time (women : ℚ) (tea : ℚ) : ℚ :=
  1.5 * tea / women

/-- Theorem stating that if 1.5 women drink 1.5 tea in 1.5 minutes,
    then 9 women can drink 18 tea in 3 minutes. -/
theorem nine_women_eighteen_tea :
  drinking_time 9 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nine_women_eighteen_tea_l4054_405445


namespace NUMINAMATH_CALUDE_total_decorations_handed_out_l4054_405442

/-- Represents the contents of a decoration box -/
structure DecorationBox where
  tinsel : Nat
  tree : Nat
  snowGlobes : Nat

/-- Calculates the total number of decorations in a box -/
def totalDecorationsPerBox (box : DecorationBox) : Nat :=
  box.tinsel + box.tree + box.snowGlobes

/-- Theorem: The total number of decorations handed out is 120 -/
theorem total_decorations_handed_out :
  let standardBox : DecorationBox := { tinsel := 4, tree := 1, snowGlobes := 5 }
  let familyBoxes : Nat := 11
  let communityBoxes : Nat := 1
  totalDecorationsPerBox standardBox * (familyBoxes + communityBoxes) = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_decorations_handed_out_l4054_405442


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4054_405421

/-- The focus of the parabola y = 3x^2 has coordinates (0, 1/12) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), y = 3 * x^2 → ∃ (p : ℝ), p > 0 ∧ x^2 = (1/(4*p)) * y ∧ (0, p) = (0, 1/12) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4054_405421


namespace NUMINAMATH_CALUDE_allison_wins_prob_l4054_405472

/-- Represents a 6-sided cube with specified face values -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube with all faces showing 6 -/
def allison_cube : Cube :=
  { faces := fun _ => 6 }

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  { faces := fun i => i.val + 1 }

/-- Noah's cube with three faces showing 3 and three faces showing 5 -/
def noah_cube : Cube :=
  { faces := fun i => if i.val < 3 then 3 else 5 }

/-- The probability of rolling a value less than n on a given cube -/
def prob_less_than (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (fun i => c.faces i < n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison's roll being greater than both Brian's and Noah's -/
theorem allison_wins_prob :
    prob_less_than brian_cube 6 * prob_less_than noah_cube 6 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_allison_wins_prob_l4054_405472


namespace NUMINAMATH_CALUDE_divisibility_by_two_l4054_405406

theorem divisibility_by_two (a b : ℕ) (h : 2 ∣ (a * b)) : ¬(¬(2 ∣ a) ∧ ¬(2 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_two_l4054_405406


namespace NUMINAMATH_CALUDE_min_value_a_l4054_405483

theorem min_value_a (a m n : ℕ) (h1 : a ≠ 0) (h2 : (2 : ℚ) / 10 * a = m ^ 2) (h3 : (5 : ℚ) / 10 * a = n ^ 3) :
  ∀ b : ℕ, b ≠ 0 ∧ (∃ p q : ℕ, (2 : ℚ) / 10 * b = p ^ 2 ∧ (5 : ℚ) / 10 * b = q ^ 3) → a ≤ b → a = 2000 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l4054_405483


namespace NUMINAMATH_CALUDE_population_scientific_notation_l4054_405434

def population : ℝ := 1411750000

theorem population_scientific_notation : 
  population = 1.41175 * (10 : ℝ) ^ 9 :=
sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l4054_405434


namespace NUMINAMATH_CALUDE_min_sum_m_n_l4054_405435

theorem min_sum_m_n (m n : ℕ+) (h : 45 * m = n^3) : 
  (∀ m' n' : ℕ+, 45 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 90 := by
sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l4054_405435


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l4054_405480

/-- Given a line and a circle intersecting at two distinct points, with a condition on the vectors,
    this theorem determines the range of the parameter m in the line equation. -/
theorem line_circle_intersection_range (m : ℝ) : 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.1 + A.2 + m = 0) ∧ (B.1 + B.2 + m = 0) ∧
   (A.1^2 + A.2^2 = 2) ∧ (B.1^2 + B.2^2 = 2) ∧
   ‖(A.1, A.2)‖ + ‖(B.1, B.2)‖ ≥ ‖(A.1 - B.1, A.2 - B.2)‖) →
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Icc (Real.sqrt 2) 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l4054_405480


namespace NUMINAMATH_CALUDE_round_table_numbers_l4054_405497

theorem round_table_numbers (n : Fin 10 → ℝ) 
  (h1 : (n 9 + n 1) / 2 = 1)
  (h2 : (n 0 + n 2) / 2 = 2)
  (h3 : (n 1 + n 3) / 2 = 3)
  (h4 : (n 2 + n 4) / 2 = 4)
  (h5 : (n 3 + n 5) / 2 = 5)
  (h6 : (n 4 + n 6) / 2 = 6)
  (h7 : (n 5 + n 7) / 2 = 7)
  (h8 : (n 6 + n 8) / 2 = 8)
  (h9 : (n 7 + n 9) / 2 = 9)
  (h10 : (n 8 + n 0) / 2 = 10) :
  n 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_round_table_numbers_l4054_405497


namespace NUMINAMATH_CALUDE_iron_to_steel_ratio_l4054_405485

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the composition of an alloy -/
structure Alloy where
  iron : ℕ
  steel : ℕ

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

/-- Theorem: The ratio of iron to steel in the alloy is 2:5 -/
theorem iron_to_steel_ratio (alloy : Alloy) (h : alloy = { iron := 14, steel := 35 }) :
  simplifyRatio { numerator := alloy.iron, denominator := alloy.steel } = { numerator := 2, denominator := 5 } := by
  sorry

end NUMINAMATH_CALUDE_iron_to_steel_ratio_l4054_405485


namespace NUMINAMATH_CALUDE_intersection_points_quadratic_linear_l4054_405481

theorem intersection_points_quadratic_linear 
  (x y : ℝ) : 
  (y = 3 * x^2 - 6 * x + 5 ∧ y = 2 * x + 1) ↔ 
  ((x = 2 ∧ y = 5) ∨ (x = 2/3 ∧ y = 7/3)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_quadratic_linear_l4054_405481


namespace NUMINAMATH_CALUDE_calculation_difference_l4054_405439

theorem calculation_difference : 
  let correct_calculation := 10 - (3 * 4)
  let incorrect_calculation := 10 - 3 + 4
  correct_calculation - incorrect_calculation = -13 := by
sorry

end NUMINAMATH_CALUDE_calculation_difference_l4054_405439


namespace NUMINAMATH_CALUDE_excellent_grade_percentage_l4054_405471

theorem excellent_grade_percentage (total : ℕ) (excellent : ℕ) (h1 : total = 360) (h2 : excellent = 72) :
  (excellent : ℚ) / (total : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_excellent_grade_percentage_l4054_405471


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_fraction_in_lowest_terms_l4054_405453

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a : ℚ) + (b * 10 + c : ℚ) / 990

theorem repeating_decimal_equiv_fraction :
  repeating_decimal_to_fraction 4 1 7 = 413 / 990 :=
sorry

theorem fraction_in_lowest_terms : ∀ n : ℕ, n > 1 → n ∣ 413 → n ∣ 990 → False :=
sorry

#eval repeating_decimal_to_fraction 4 1 7

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_fraction_in_lowest_terms_l4054_405453


namespace NUMINAMATH_CALUDE_largest_in_systematic_sample_l4054_405404

/-- Represents a systematic sample --/
structure SystematicSample where
  total : Nat
  start : Nat
  interval : Nat

/-- Checks if a number is in the systematic sample --/
def inSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.start + k * s.interval ∧ n ≤ s.total

/-- The largest number in the sample --/
def largestInSample (s : SystematicSample) : Nat :=
  s.start + ((s.total - s.start) / s.interval) * s.interval

theorem largest_in_systematic_sample
  (employees : Nat)
  (first : Nat)
  (second : Nat)
  (h1 : employees = 500)
  (h2 : first = 6)
  (h3 : second = 31)
  (h4 : second - first = 31 - 6) :
  let s := SystematicSample.mk employees first (second - first)
  largestInSample s = 481 :=
by
  sorry

#check largest_in_systematic_sample

end NUMINAMATH_CALUDE_largest_in_systematic_sample_l4054_405404


namespace NUMINAMATH_CALUDE_concert_expense_l4054_405401

def ticket_price : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem concert_expense : 
  ticket_price * (tickets_for_friends + extra_tickets) = 60 := by
  sorry

end NUMINAMATH_CALUDE_concert_expense_l4054_405401


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l4054_405450

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l4054_405450


namespace NUMINAMATH_CALUDE_arrangements_count_l4054_405443

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 2

/-- The number of girls -/
def num_girls : ℕ := 3

/-- A function that calculates the number of arrangements -/
def count_arrangements (n : ℕ) (b : ℕ) (g : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem arrangements_count :
  count_arrangements total_students num_boys num_girls = 48 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l4054_405443


namespace NUMINAMATH_CALUDE_cone_slant_height_l4054_405487

/-- The slant height of a cone given its base radius and curved surface area -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 10) (h2 : csa = 628.3185307179587) :
  csa / (π * r) = 20 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l4054_405487


namespace NUMINAMATH_CALUDE_plane_relationships_l4054_405456

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem plane_relationships 
  (α β : Plane) 
  (h_different : α ≠ β) :
  (∀ l : Line, in_plane l α → 
    (∀ m : Line, in_plane m β → perpendicular l m) → 
    plane_perpendicular α β) ∧
  ((∀ l : Line, in_plane l α → line_parallel_to_plane l β) → 
    plane_parallel α β) ∧
  (plane_parallel α β → 
    ∀ l : Line, in_plane l α → line_parallel_to_plane l β) :=
sorry

end NUMINAMATH_CALUDE_plane_relationships_l4054_405456


namespace NUMINAMATH_CALUDE_jessica_and_sibling_ages_l4054_405469

-- Define the variables
def jessica_age_at_passing : ℕ := sorry
def mother_age_at_passing : ℕ := sorry
def current_year : ℕ := sorry
def sibling_age : ℕ := sorry

-- Define the conditions
def jessica_half_mother_age : Prop :=
  jessica_age_at_passing = mother_age_at_passing / 2

def mother_age_if_alive : Prop :=
  mother_age_at_passing + 10 = 70

def sibling_age_difference : Prop :=
  sibling_age - (jessica_age_at_passing + 10) = (70 - mother_age_at_passing) / 2

-- Theorem to prove
theorem jessica_and_sibling_ages :
  jessica_half_mother_age →
  mother_age_if_alive →
  sibling_age_difference →
  jessica_age_at_passing + 10 = 40 ∧ sibling_age = 45 := by
  sorry

end NUMINAMATH_CALUDE_jessica_and_sibling_ages_l4054_405469


namespace NUMINAMATH_CALUDE_margaret_permutation_time_l4054_405478

/-- Calculates the time required to write all permutations of a name -/
def time_to_write_permutations (name_length : ℕ) (permutations_per_minute : ℕ) : ℕ × ℕ :=
  let total_permutations := Nat.factorial name_length
  let total_minutes := total_permutations / permutations_per_minute
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

/-- Theorem: Given Margaret's name length and writing speed, the time to write all permutations is 44 hours and 48 minutes -/
theorem margaret_permutation_time :
  time_to_write_permutations 8 15 = (44, 48) := by
  sorry

end NUMINAMATH_CALUDE_margaret_permutation_time_l4054_405478


namespace NUMINAMATH_CALUDE_min_cost_is_800_l4054_405489

/-- Represents the number of adults in the group -/
def num_adults : ℕ := 8

/-- Represents the number of children in the group -/
def num_children : ℕ := 4

/-- Represents the cost of an adult ticket in yuan -/
def adult_ticket_cost : ℕ := 100

/-- Represents the cost of a child ticket in yuan -/
def child_ticket_cost : ℕ := 50

/-- Represents the cost of a group ticket per person in yuan -/
def group_ticket_cost : ℕ := 70

/-- Represents the minimum number of people required for group tickets -/
def min_group_size : ℕ := 10

/-- Calculates the total cost of tickets given the number of group tickets and individual tickets -/
def total_cost (num_group : ℕ) (num_individual_adult : ℕ) (num_individual_child : ℕ) : ℕ :=
  num_group * group_ticket_cost + 
  num_individual_adult * adult_ticket_cost + 
  num_individual_child * child_ticket_cost

/-- Theorem stating that the minimum cost for the given group is 800 yuan -/
theorem min_cost_is_800 : 
  ∀ (num_group : ℕ) (num_individual_adult : ℕ) (num_individual_child : ℕ),
    num_group + num_individual_adult + num_individual_child = num_adults + num_children →
    num_group = 0 ∨ num_group ≥ min_group_size →
    total_cost num_group num_individual_adult num_individual_child ≥ 800 ∧
    (∃ (ng na nc : ℕ), 
      ng + na + nc = num_adults + num_children ∧
      (ng = 0 ∨ ng ≥ min_group_size) ∧
      total_cost ng na nc = 800) := by
  sorry

#check min_cost_is_800

end NUMINAMATH_CALUDE_min_cost_is_800_l4054_405489


namespace NUMINAMATH_CALUDE_work_completion_time_l4054_405440

/-- Proves that given 15 original workers and 10 additional workers, 
    if the work is finished 3 days earlier with the additional workers, 
    then the original time to complete the work without additional workers is 8 days. -/
theorem work_completion_time 
  (original_workers : ℕ) 
  (additional_workers : ℕ) 
  (time_saved : ℕ) 
  (h1 : original_workers = 15)
  (h2 : additional_workers = 10)
  (h3 : time_saved = 3) :
  ∃ (original_time : ℕ), 
    original_time * original_workers = 
      (original_time - time_saved) * (original_workers + additional_workers) ∧
    original_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4054_405440


namespace NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l4054_405495

/-- Conversion of 15 degrees to radians -/
theorem fifteen_degrees_to_radians : 
  (15 : ℝ) * π / 180 = π / 12 := by sorry

end NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l4054_405495


namespace NUMINAMATH_CALUDE_mod_nine_equivalence_l4054_405454

theorem mod_nine_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_mod_nine_equivalence_l4054_405454


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l4054_405496

theorem sqrt_2x_minus_1_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 1) ↔ x ≥ (1/2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l4054_405496


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l4054_405482

theorem diophantine_equation_solutions :
  ∀ x y z w : ℕ,
  2^x * 3^y - 5^z * 7^w = 1 ↔
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l4054_405482


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l4054_405459

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 2*a - 3 = 0) → (b^2 - 2*b - 3 = 0) → (a ≠ b) → a^2 + b^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l4054_405459


namespace NUMINAMATH_CALUDE_cos_tan_arcsin_four_fifths_l4054_405468

theorem cos_tan_arcsin_four_fifths :
  (∃ θ : ℝ, θ = Real.arcsin (4/5)) →
  (Real.cos (Real.arcsin (4/5)) = 3/5) ∧
  (Real.tan (Real.arcsin (4/5)) = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_cos_tan_arcsin_four_fifths_l4054_405468


namespace NUMINAMATH_CALUDE_frank_game_points_l4054_405474

theorem frank_game_points (enemies_defeated : ℕ) (points_per_enemy : ℕ) 
  (level_completion_points : ℕ) (special_challenges : ℕ) (points_per_challenge : ℕ) : 
  enemies_defeated = 15 → 
  points_per_enemy = 12 → 
  level_completion_points = 20 → 
  special_challenges = 5 → 
  points_per_challenge = 10 → 
  enemies_defeated * points_per_enemy + level_completion_points + special_challenges * points_per_challenge = 250 := by
  sorry

#check frank_game_points

end NUMINAMATH_CALUDE_frank_game_points_l4054_405474


namespace NUMINAMATH_CALUDE_simplify_expression_l4054_405467

theorem simplify_expression (a : ℝ) : 
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4054_405467


namespace NUMINAMATH_CALUDE_exists_n_sum_digits_decreases_l4054_405428

-- Define the sum of digits function
def S (a : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_n_sum_digits_decreases :
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n+1)) := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_digits_decreases_l4054_405428


namespace NUMINAMATH_CALUDE_sum_50_to_75_l4054_405465

/-- Sum of integers from a to b, inclusive -/
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

/-- Theorem: The sum of all integers from 50 through 75, inclusive, is 1625 -/
theorem sum_50_to_75 : sum_integers 50 75 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_sum_50_to_75_l4054_405465


namespace NUMINAMATH_CALUDE_ben_dogs_difference_l4054_405408

/-- The number of dogs Teddy has -/
def teddy_dogs : ℕ := 7

/-- The number of cats Teddy has -/
def teddy_cats : ℕ := 8

/-- The number of cats Dave has -/
def dave_cats : ℕ := teddy_cats + 13

/-- The number of dogs Dave has -/
def dave_dogs : ℕ := teddy_dogs - 5

/-- The total number of pets all three have -/
def total_pets : ℕ := 54

/-- The number of dogs Ben has -/
def ben_dogs : ℕ := total_pets - (teddy_dogs + teddy_cats + dave_dogs + dave_cats)

theorem ben_dogs_difference : ben_dogs - teddy_dogs = 9 := by
  sorry

end NUMINAMATH_CALUDE_ben_dogs_difference_l4054_405408


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l4054_405446

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.orange_percent = 0.25)
  (h2 : drink.watermelon_percent = 0.40)
  (h3 : drink.grape_ounces = 70)
  (h4 : drink.orange_percent + drink.watermelon_percent + drink.grape_ounces / drink.total = 1) :
  drink.total = 200 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l4054_405446


namespace NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l4054_405427

theorem cinnamon_swirl_sharing (total_swirls : ℕ) (jane_pieces : ℕ) (people : ℕ) : 
  total_swirls = 12 →
  jane_pieces = 4 →
  total_swirls % jane_pieces = 0 →
  total_swirls / jane_pieces = people →
  people = 3 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l4054_405427


namespace NUMINAMATH_CALUDE_periodic_properties_l4054_405415

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ T > 0, ∀ x, f (x + T) = f x

-- Define a non-periodic function
def NonPeriodic (g : ℝ → ℝ) : Prop :=
  ¬ Periodic g

theorem periodic_properties
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : Periodic f) (hg : NonPeriodic g) :
  Periodic (fun x ↦ (f x)^2) ∧
  NonPeriodic (fun x ↦ Real.sqrt (g x)) ∧
  Periodic (g ∘ f) :=
sorry

end NUMINAMATH_CALUDE_periodic_properties_l4054_405415


namespace NUMINAMATH_CALUDE_infinite_factorial_solutions_l4054_405455

theorem infinite_factorial_solutions :
  ∃ f : ℕ → ℕ × ℕ × ℕ, ∀ n : ℕ,
    let (x, y, z) := f n
    x > 1 ∧ y > 1 ∧ z > 1 ∧ Nat.factorial x * Nat.factorial y = Nat.factorial z :=
by sorry

end NUMINAMATH_CALUDE_infinite_factorial_solutions_l4054_405455


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l4054_405476

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l4054_405476


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l4054_405464

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9

theorem ferris_wheel_cost : (tickets_bought - tickets_left) * ticket_cost = 81 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l4054_405464


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l4054_405463

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability 
  (group1_size : ℕ) 
  (group2_size : ℕ) 
  (p : ℝ) 
  (h1 : group1_size = 6) 
  (h2 : group2_size = 7) 
  (h3 : 0 ≤ p ∧ p ≤ 1) : 
  contact_probability p = 1 - (1 - p) ^ (group1_size * group2_size) :=
by sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l4054_405463


namespace NUMINAMATH_CALUDE_english_only_enrollment_l4054_405441

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 60)
  (h2 : both = 18)
  (h3 : german = 36)
  (h4 : total ≥ german)
  (h5 : german ≥ both) :
  total - (german - both) - both = 24 :=
by sorry

end NUMINAMATH_CALUDE_english_only_enrollment_l4054_405441


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l4054_405400

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- The first line: (k-1)x + y + 2 = 0 -/
def line1 (k x y : ℝ) : Prop :=
  (k - 1) * x + y + 2 = 0

/-- The second line: 8x + (k+1)y + k - 1 = 0 -/
def line2 (k x y : ℝ) : Prop :=
  8 * x + (k + 1) * y + k - 1 = 0

theorem parallel_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, are_parallel (k - 1) 1 8 (k + 1)) →
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l4054_405400


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l4054_405491

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 2 → 
  k > 0 → 
  a * b ≠ 0 → 
  a = (k + 1) * b → 
  (n.choose 1 * (k * b)^(n - 1) * (-b) + n.choose 2 * (k * b)^(n - 2) * (-b)^2 = k * b^n * k^(n - 2)) → 
  n = 2 * k + 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l4054_405491


namespace NUMINAMATH_CALUDE_money_redistribution_l4054_405437

def initial_amount (i : Nat) : Nat :=
  2^(i-1) - 1

def final_amount (n : Nat) : Nat :=
  8 * (List.sum (List.map initial_amount (List.range n)))

theorem money_redistribution (n : Nat) :
  n = 9 → final_amount n = 512 := by sorry

end NUMINAMATH_CALUDE_money_redistribution_l4054_405437


namespace NUMINAMATH_CALUDE_inverse_of_matrix_A_l4054_405414

theorem inverse_of_matrix_A (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A * !![1, 2; 0, 6] = !![(-1), (-2); 0, 3] →
  A⁻¹ = !![(-1), 0; 0, 2] := by sorry

end NUMINAMATH_CALUDE_inverse_of_matrix_A_l4054_405414


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l4054_405405

theorem sine_cosine_relation (α : Real) (h : Real.sin (α + π/6) = 1/3) : 
  Real.cos (α - π/3) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l4054_405405


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l4054_405407

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (purchase_price : ℝ)
  (price_10 : ℝ)
  (sales_10 : ℝ)
  (price_13 : ℝ)
  (profit_13 : ℝ)
  (h1 : purchase_price = 8)
  (h2 : price_10 = 10)
  (h3 : sales_10 = 300)
  (h4 : price_13 = 13)
  (h5 : profit_13 = 750)
  (y : ℝ → ℝ)
  (h6 : ∀ x > 0, ∃ k b : ℝ, y x = k * x + b) :
  (∃ k b : ℝ, ∀ x > 0, y x = k * x + b ∧ k = -50 ∧ b = 800) ∧
  (∃ max_price : ℝ, max_price = 12 ∧ 
    ∀ x > 0, (y x) * (x - purchase_price) ≤ (y max_price) * (max_price - purchase_price)) ∧
  (∃ max_profit : ℝ, max_profit = 800 ∧
    max_profit = (y 12) * (12 - purchase_price)) :=
by sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l4054_405407


namespace NUMINAMATH_CALUDE_rotation_and_inclination_l4054_405473

/-- Given a point A(2,1) rotated counterclockwise around the origin O by π/4 to point B,
    if the angle of inclination of line OB is α, then cos α = √10/10 -/
theorem rotation_and_inclination :
  let A : ℝ × ℝ := (2, 1)
  let rotation_angle : ℝ := π / 4
  let B : ℝ × ℝ := (
    A.1 * Real.cos rotation_angle - A.2 * Real.sin rotation_angle,
    A.1 * Real.sin rotation_angle + A.2 * Real.cos rotation_angle
  )
  let α : ℝ := Real.arctan (B.2 / B.1)
  Real.cos α = Real.sqrt 10 / 10 := by sorry

end NUMINAMATH_CALUDE_rotation_and_inclination_l4054_405473


namespace NUMINAMATH_CALUDE_sugar_solution_sweetness_l4054_405494

theorem sugar_solution_sweetness (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 0) (hab : a > b) :
  (b + t) / (a + t) > b / a :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_sweetness_l4054_405494


namespace NUMINAMATH_CALUDE_range_of_k_for_decreasing_proportional_function_l4054_405457

/-- A proportional function y = (k+4)x where y decreases as x increases -/
def decreasing_proportional_function (k : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k + 4) * x₁ > (k + 4) * x₂

/-- The range of k for a decreasing proportional function y = (k+4)x -/
theorem range_of_k_for_decreasing_proportional_function :
  ∀ k : ℝ, decreasing_proportional_function k → k < -4 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_decreasing_proportional_function_l4054_405457


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l4054_405402

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) :
  ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l4054_405402


namespace NUMINAMATH_CALUDE_tank_emptying_time_l4054_405448

/-- Given a tank with specified capacity, leak rate, and inlet rate, 
    calculate the time it takes to empty when both leak and inlet are open. -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 1440) 
  (h2 : leak_time = 3) 
  (h3 : inlet_rate_per_minute = 6) : 
  (tank_capacity / (tank_capacity / leak_time - inlet_rate_per_minute * 60)) = 12 :=
by
  sorry

#check tank_emptying_time

end NUMINAMATH_CALUDE_tank_emptying_time_l4054_405448


namespace NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l4054_405498

theorem cal_anthony_transaction_ratio :
  ∀ (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ),
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    jade_transactions = 84 →
    jade_transactions = cal_transactions + 18 →
    cal_transactions * 3 = anthony_transactions * 2 := by
  sorry

end NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l4054_405498


namespace NUMINAMATH_CALUDE_two_color_line_exists_l4054_405416

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- Represents a point in the 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := Point → Color

/-- Predicate to check if four points form a unit square -/
def isUnitSquare (p1 p2 p3 p4 : Point) : Prop :=
  (p1.x = p2.x ∧ p1.y + 1 = p2.y) ∧
  (p2.x + 1 = p3.x ∧ p2.y = p3.y) ∧
  (p3.x = p4.x ∧ p3.y - 1 = p4.y) ∧
  (p4.x - 1 = p1.x ∧ p4.y = p1.y)

/-- Predicate to check if a coloring is valid (adjacent nodes in unit squares have different colors) -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ p1 p2 p3 p4 : Point, isUnitSquare p1 p2 p3 p4 →
    c p1 ≠ c p2 ∧ c p1 ≠ c p3 ∧ c p1 ≠ c p4 ∧
    c p2 ≠ c p3 ∧ c p2 ≠ c p4 ∧
    c p3 ≠ c p4

/-- Predicate to check if a line (horizontal or vertical) uses only two colors -/
def lineUsesTwoColors (c : Coloring) : Prop :=
  (∃ y : ℤ, ∃ c1 c2 : Color, ∀ x : ℤ, c ⟨x, y⟩ = c1 ∨ c ⟨x, y⟩ = c2) ∨
  (∃ x : ℤ, ∃ c1 c2 : Color, ∀ y : ℤ, c ⟨x, y⟩ = c1 ∨ c ⟨x, y⟩ = c2)

theorem two_color_line_exists (c : Coloring) (h : isValidColoring c) : lineUsesTwoColors c := by
  sorry

end NUMINAMATH_CALUDE_two_color_line_exists_l4054_405416


namespace NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_radius_l4054_405426

theorem right_triangle_circumscribed_circle_radius 
  (a b c R : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 5) 
  (h_b : b = 12) 
  (h_R : R = c / 2) : R = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_radius_l4054_405426


namespace NUMINAMATH_CALUDE_prob_no_consecutive_ones_l4054_405451

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of binary sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- Probability of no consecutive 1s in a sequence of length n -/
def prob (n : ℕ) : ℚ := (validSequences n : ℚ) / (totalSequences n : ℚ)

theorem prob_no_consecutive_ones : prob 12 = 377 / 4096 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_ones_l4054_405451


namespace NUMINAMATH_CALUDE_jims_taxi_additional_charge_l4054_405486

/-- The additional charge for each 2/5 of a mile in Jim's taxi service -/
def additional_charge (initial_fee total_distance total_charge : ℚ) : ℚ :=
  ((total_charge - initial_fee) * 2) / (5 * total_distance)

/-- Theorem stating the additional charge for each 2/5 of a mile in Jim's taxi service -/
theorem jims_taxi_additional_charge :
  additional_charge (5/2) (36/10) (565/100) = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_jims_taxi_additional_charge_l4054_405486


namespace NUMINAMATH_CALUDE_quadratic_roots_l4054_405461

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_roots (b c : ℝ) :
  (f b c (-2) = 5) →
  (f b c (-1) = 0) →
  (f b c 0 = -3) →
  (f b c 1 = -4) →
  (f b c 2 = -3) →
  (f b c 4 = 5) →
  (∃ x, f b c x = 0) →
  (∀ x, f b c x = 0 ↔ (x = -1 ∨ x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l4054_405461
