import Mathlib

namespace smallest_number_of_eggs_smallest_number_of_eggs_proof_l2273_227318

theorem smallest_number_of_eggs : ℕ → Prop :=
  fun n =>
    (n > 100) ∧
    (∃ c : ℕ, n = 15 * c - 3) ∧
    (∀ m : ℕ, m > 100 ∧ (∃ d : ℕ, m = 15 * d - 3) → m ≥ n) →
    n = 102

-- The proof goes here
theorem smallest_number_of_eggs_proof : smallest_number_of_eggs 102 := by
  sorry

end smallest_number_of_eggs_smallest_number_of_eggs_proof_l2273_227318


namespace regular_nonagon_side_length_l2273_227378

/-- A regular nonagon with perimeter 171 centimeters has sides of length 19 centimeters -/
theorem regular_nonagon_side_length : 
  ∀ (perimeter side_length : ℝ),
    perimeter = 171 →
    side_length * 9 = perimeter →
    side_length = 19 :=
by sorry

end regular_nonagon_side_length_l2273_227378


namespace chess_tournament_games_l2273_227373

/-- The number of games played in a round-robin tournament -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 15 players, where each player plays every other player exactly once,
    and each game is played by two players, the total number of games played is 105. -/
theorem chess_tournament_games :
  gamesPlayed 15 = 105 := by
  sorry

#eval gamesPlayed 15  -- This will evaluate to 105

end chess_tournament_games_l2273_227373


namespace last_digit_of_power_difference_l2273_227348

theorem last_digit_of_power_difference : (7^95 - 3^58) % 10 = 4 := by
  sorry

end last_digit_of_power_difference_l2273_227348


namespace jacob_age_problem_l2273_227344

theorem jacob_age_problem (x : ℕ) : 
  (40 + x : ℕ) = 3 * (10 + x) ∧ 
  (40 - x : ℕ) = 7 * (10 - x) → 
  x = 5 := by sorry

end jacob_age_problem_l2273_227344


namespace volume_of_specific_prism_l2273_227334

/-- A regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Length of AD, where D is on the diameter CD -/
  AD : ℝ

/-- The volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ := sorry

/-- Theorem: The volume of the specific inscribed prism is 48√15 -/
theorem volume_of_specific_prism :
  let p : InscribedPrism := { R := 6, AD := 4 * Real.sqrt 6 }
  prism_volume p = 48 * Real.sqrt 15 := by sorry

end volume_of_specific_prism_l2273_227334


namespace total_monthly_payment_l2273_227325

def basic_service : ℕ := 15
def movie_channels : ℕ := 12
def sports_channels : ℕ := movie_channels - 3

theorem total_monthly_payment :
  basic_service + movie_channels + sports_channels = 36 :=
by sorry

end total_monthly_payment_l2273_227325


namespace floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1_l2273_227398

theorem floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1 : 
  Int.floor (Real.sqrt 45)^2 + 2 * Int.floor (Real.sqrt 45) + 1 = 49 := by sorry

end floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1_l2273_227398


namespace melted_ice_cream_height_l2273_227305

/-- The height of a cylinder resulting from a melted sphere, given constant volume --/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3)
  (h_cylinder : r_cylinder = 10) :
  (4 / 3 * π * r_sphere ^ 3) / (π * r_cylinder ^ 2) = 9 / 25 := by
  sorry

end melted_ice_cream_height_l2273_227305


namespace direction_vector_b_l2273_227393

/-- Prove that for a line passing through points (-6, 0) and (-3, 3), its direction vector (3, b) has b = 3. -/
theorem direction_vector_b (b : ℝ) : 
  let p1 : ℝ × ℝ := (-6, 0)
  let p2 : ℝ × ℝ := (-3, 3)
  let direction_vector : ℝ × ℝ := (3, b)
  (p2.1 - p1.1 = direction_vector.1 ∧ p2.2 - p1.2 = direction_vector.2) → b = 3 := by
  sorry

end direction_vector_b_l2273_227393


namespace video_recorder_markup_percentage_l2273_227388

/-- Proves that the percentage markup over wholesale cost is 20% for a video recorder --/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_discount : ℝ)
  (employee_paid : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_discount = 0.05)
  (h3 : employee_paid = 228)
  : ∃ (markup_percentage : ℝ),
    markup_percentage = 20 ∧
    employee_paid = (1 - employee_discount) * (wholesale_cost * (1 + markup_percentage / 100)) :=
by sorry

end video_recorder_markup_percentage_l2273_227388


namespace boat_speed_l2273_227366

/-- Proves that the speed of a boat in still water is 30 kmph given specific conditions -/
theorem boat_speed (x : ℝ) (h1 : x > 0) : 
  (∃ t : ℝ, t > 0 ∧ 80 = (x + 10) * t ∧ 40 = (x - 10) * t) → x = 30 := by
  sorry

#check boat_speed

end boat_speed_l2273_227366


namespace compute_expression_l2273_227356

theorem compute_expression : 8 * (2/3)^4 + 2 = 290/81 := by
  sorry

end compute_expression_l2273_227356


namespace max_candies_karlson_candy_theorem_l2273_227360

/-- Represents the process of combining numbers and counting products -/
def combine_numbers (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- The maximum number of candies Karlson can eat -/
theorem max_candies : combine_numbers 26 = 325 := by
  sorry

/-- Proves that the maximum number of candies is achieved -/
theorem karlson_candy_theorem (initial_count : ℕ) (operation_count : ℕ) 
  (h1 : initial_count = 26) (h2 : operation_count = 25) : 
  combine_numbers initial_count = 325 := by
  sorry

end max_candies_karlson_candy_theorem_l2273_227360


namespace prob_one_girl_l2273_227308

theorem prob_one_girl (p_two_boys p_two_girls : ℚ) 
  (h1 : p_two_boys = 1/3) 
  (h2 : p_two_girls = 2/15) : 
  1 - (p_two_boys + p_two_girls) = 8/15 := by
  sorry

end prob_one_girl_l2273_227308


namespace inscribed_trapezoid_leg_length_l2273_227381

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  radius : ℝ
  base1 : ℝ
  base2 : ℝ
  centerInside : Bool

/-- The average length of the legs of the trapezoid squared -/
def averageLegLengthSquared (t : InscribedTrapezoid) : ℝ :=
  sorry

/-- Theorem: For a trapezoid JANE inscribed in a circle of radius 25 with the center inside,
    if the bases are 14 and 30, then the average leg length squared is 2000 -/
theorem inscribed_trapezoid_leg_length
    (t : InscribedTrapezoid)
    (h1 : t.radius = 25)
    (h2 : t.base1 = 14)
    (h3 : t.base2 = 30)
    (h4 : t.centerInside = true) :
  averageLegLengthSquared t = 2000 := by
  sorry

end inscribed_trapezoid_leg_length_l2273_227381


namespace piglet_banana_count_l2273_227320

/-- Represents the number of bananas eaten by each character -/
structure BananaCount where
  winnie : ℕ
  owl : ℕ
  rabbit : ℕ
  piglet : ℕ

/-- The conditions of the banana distribution problem -/
def BananaDistribution (bc : BananaCount) : Prop :=
  bc.winnie + bc.owl + bc.rabbit + bc.piglet = 70 ∧
  bc.owl + bc.rabbit = 45 ∧
  bc.winnie > bc.owl ∧
  bc.winnie > bc.rabbit ∧
  bc.winnie > bc.piglet ∧
  bc.winnie ≥ 1 ∧
  bc.owl ≥ 1 ∧
  bc.rabbit ≥ 1 ∧
  bc.piglet ≥ 1

theorem piglet_banana_count (bc : BananaCount) :
  BananaDistribution bc → bc.piglet = 1 := by
  sorry

end piglet_banana_count_l2273_227320


namespace sin_neg_seven_pi_thirds_l2273_227390

theorem sin_neg_seven_pi_thirds : Real.sin (-7 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_neg_seven_pi_thirds_l2273_227390


namespace quadratic_equation_roots_average_l2273_227311

theorem quadratic_equation_roots_average (a b : ℝ) (h : a ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ 3*a*x^2 - 6*a*x + 2*b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 1 := by
sorry

end quadratic_equation_roots_average_l2273_227311


namespace triangle_angle_measure_l2273_227316

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  a = 2 * Real.sin A →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  A + B + C = π →
  A = π / 6 := by
sorry

end triangle_angle_measure_l2273_227316


namespace cookie_price_calculation_l2273_227368

def trip_cost : ℝ := 5000
def hourly_wage : ℝ := 20
def hours_worked : ℝ := 10
def cookies_sold : ℝ := 24
def lottery_ticket_cost : ℝ := 10
def lottery_winnings : ℝ := 500
def gift_per_sister : ℝ := 500
def num_sisters : ℝ := 2
def additional_money_needed : ℝ := 3214

theorem cookie_price_calculation (trip_cost hourly_wage hours_worked 
  cookies_sold lottery_ticket_cost lottery_winnings gift_per_sister 
  num_sisters additional_money_needed : ℝ) :
  let total_earnings := hourly_wage * hours_worked + 
    lottery_winnings + gift_per_sister * num_sisters - lottery_ticket_cost
  let cookie_revenue := trip_cost - total_earnings
  cookie_revenue / cookies_sold = 204.33 := by
  sorry

end cookie_price_calculation_l2273_227368


namespace abc_inequality_l2273_227328

theorem abc_inequality (a b c : ℝ) (ha : |a| < 1) (hb : |b| < 1) (hc : |c| < 1) :
  a * b + b * c + c * a > -1 := by
  sorry

end abc_inequality_l2273_227328


namespace sin_cos_identity_l2273_227341

theorem sin_cos_identity : 
  Real.sin (10 * π / 180) * Real.cos (70 * π / 180) - 
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_identity_l2273_227341


namespace max_y_over_x_l2273_227391

theorem max_y_over_x (x y : ℝ) (h : x^2 + y^2 - 6*x - 6*y + 12 = 0) :
  ∃ (k : ℝ), k = 3 + 2 * Real.sqrt 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 6*x' - 6*y' + 12 = 0 → y' / x' ≤ k := by
  sorry

end max_y_over_x_l2273_227391


namespace classroom_students_l2273_227359

theorem classroom_students (n : ℕ) : 
  n < 50 → n % 8 = 5 → n % 6 = 3 → (n = 21 ∨ n = 45) := by
  sorry

end classroom_students_l2273_227359


namespace scientific_notation_308000000_l2273_227301

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_308000000 :
  to_scientific_notation 308000000 = ScientificNotation.mk 3.08 8 (by sorry) :=
sorry

end scientific_notation_308000000_l2273_227301


namespace additional_amount_needed_l2273_227322

def pencil_cost : ℚ := 6
def notebook_cost : ℚ := 7/2
def pen_cost : ℚ := 9/4
def initial_amount : ℚ := 5
def borrowed_amount : ℚ := 53/100

def total_cost : ℚ := pencil_cost + notebook_cost + pen_cost
def total_available : ℚ := initial_amount + borrowed_amount

theorem additional_amount_needed : total_cost - total_available = 311/50 := by
  sorry

end additional_amount_needed_l2273_227322


namespace quadratic_inequality_solution_set_l2273_227312

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end quadratic_inequality_solution_set_l2273_227312


namespace rescue_center_dog_count_l2273_227389

/-- Calculates the final number of dogs in an animal rescue center after a series of events. -/
def final_dog_count (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that given specific values for initial count, dogs moved in, and adoptions,
    the final count of dogs is 200. -/
theorem rescue_center_dog_count :
  final_dog_count 200 100 40 60 = 200 := by
  sorry

#eval final_dog_count 200 100 40 60

end rescue_center_dog_count_l2273_227389


namespace jia_age_is_24_l2273_227362

/-- Represents the ages of four individuals -/
structure Ages where
  jia : ℕ
  yi : ℕ
  bing : ℕ
  ding : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- Jia is 4 years older than Yi
  ages.jia = ages.yi + 4 ∧
  -- Ding is 17 years old
  ages.ding = 17 ∧
  -- The average age of Jia, Yi, and Bing is 1 year more than the average age of all four people
  (ages.jia + ages.yi + ages.bing) / 3 = (ages.jia + ages.yi + ages.bing + ages.ding) / 4 + 1 ∧
  -- The average age of Jia and Yi is 1 year more than the average age of Jia, Yi, and Bing
  (ages.jia + ages.yi) / 2 = (ages.jia + ages.yi + ages.bing) / 3 + 1

/-- The theorem stating that if the conditions are satisfied, Jia's age is 24 -/
theorem jia_age_is_24 (ages : Ages) (h : satisfies_conditions ages) : ages.jia = 24 := by
  sorry

end jia_age_is_24_l2273_227362


namespace divisibility_property_l2273_227351

theorem divisibility_property (a b c d m x y : ℤ) 
  (h1 : m = a * d - b * c)
  (h2 : Nat.gcd a.natAbs m.natAbs = 1)
  (h3 : Nat.gcd b.natAbs m.natAbs = 1)
  (h4 : Nat.gcd c.natAbs m.natAbs = 1)
  (h5 : Nat.gcd d.natAbs m.natAbs = 1)
  (h6 : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) := by
  sorry

end divisibility_property_l2273_227351


namespace decomposition_count_l2273_227303

theorem decomposition_count (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (c d : ℕ), (c, d) ∈ s ↔ 
      c * d = p^2 * q^2 ∧ 
      c < d ∧ 
      d < p * q) := by sorry

end decomposition_count_l2273_227303


namespace fold_lines_cover_outside_l2273_227384

/-- A circle with center O and radius R -/
structure Circle where
  O : ℝ × ℝ
  R : ℝ

/-- A point A inside the circle -/
structure InnerPoint (c : Circle) where
  A : ℝ × ℝ
  dist_OA : Real.sqrt ((A.1 - c.O.1)^2 + (A.2 - c.O.2)^2) < c.R

/-- A point on the circumference of the circle -/
def CircumferencePoint (c : Circle) : Type :=
  { p : ℝ × ℝ // Real.sqrt ((p.1 - c.O.1)^2 + (p.2 - c.O.2)^2) = c.R }

/-- The set of all points on a fold line -/
def FoldLine (c : Circle) (A : InnerPoint c) (A' : CircumferencePoint c) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A.A + t • A'.val }

/-- The set of all points on all possible fold lines -/
def AllFoldLines (c : Circle) (A : InnerPoint c) : Set (ℝ × ℝ) :=
  ⋃ (A' : CircumferencePoint c), FoldLine c A A'

/-- The set of points outside and on the circle -/
def OutsideAndOnCircle (c : Circle) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | Real.sqrt ((p.1 - c.O.1)^2 + (p.2 - c.O.2)^2) ≥ c.R }

/-- The main theorem -/
theorem fold_lines_cover_outside (c : Circle) (A : InnerPoint c) :
  AllFoldLines c A = OutsideAndOnCircle c := by sorry

end fold_lines_cover_outside_l2273_227384


namespace inverse_proposition_l2273_227339

theorem inverse_proposition : 
  (∀ a b : ℝ, (a + b = 2 → ¬(a < 1 ∧ b < 1))) ↔ 
  (∀ a b : ℝ, (a < 1 ∧ b < 1 → a + b ≠ 2)) :=
by sorry

end inverse_proposition_l2273_227339


namespace solution_set_implies_a_values_l2273_227392

theorem solution_set_implies_a_values (a : ℕ) 
  (h : ∀ x : ℝ, (a - 2 : ℝ) * x > (a - 2 : ℝ) ↔ x < 1) : 
  a = 0 ∨ a = 1 := by
sorry

end solution_set_implies_a_values_l2273_227392


namespace complex_arithmetic_expression_l2273_227369

theorem complex_arithmetic_expression : 
  (2*(3*(2*(3*(2*(3 * (2+1) * 2)+2)*2)+2)*2)+2) = 5498 := by
  sorry

end complex_arithmetic_expression_l2273_227369


namespace stock_value_decrease_l2273_227376

theorem stock_value_decrease (n : ℕ) (n_pos : 0 < n) : (0.99 : ℝ) ^ n < 1 := by
  sorry

#check stock_value_decrease

end stock_value_decrease_l2273_227376


namespace proportional_segments_l2273_227346

/-- A set of four line segments (a, b, c, d) is proportional if a * d = b * c -/
def isProportional (a b c d : ℝ) : Prop := a * d = b * c

/-- The set of line segments (2, 4, 8, 16) is proportional -/
theorem proportional_segments : isProportional 2 4 8 16 := by
  sorry

end proportional_segments_l2273_227346


namespace solution_uniqueness_l2273_227352

theorem solution_uniqueness (x y z : ℝ) :
  x^2 * y + y^2 * z + z^2 = 0 ∧
  z^3 + z^2 * y + z * y^3 + x^2 * y = 1/4 * (x^4 + y^4) →
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end solution_uniqueness_l2273_227352


namespace complex_magnitude_equality_l2273_227307

theorem complex_magnitude_equality (s : ℝ) (hs : s > 0) :
  Complex.abs (-3 + s * Complex.I) = 2 * Real.sqrt 10 → s = Real.sqrt 31 := by
  sorry

end complex_magnitude_equality_l2273_227307


namespace prime_equation_solutions_l2273_227363

/-- A prime number (not necessarily positive) -/
def IsPrime (n : ℤ) : Prop := n ≠ 0 ∧ n ≠ 1 ∧ n ≠ -1 ∧ ∀ m : ℤ, m ∣ n → (m = 1 ∨ m = -1 ∨ m = n ∨ m = -n)

/-- The set of solutions -/
def SolutionSet : Set (ℤ × ℤ × ℤ) :=
  {(5, 2, 2), (-5, -2, -2), (-5, 3, -2), (-5, -2, 3), (5, 2, -3), (5, -3, 2)}

theorem prime_equation_solutions :
  ∀ p q r : ℤ,
    IsPrime p ∧ IsPrime q ∧ IsPrime r →
    (1 / (p - q - r : ℚ) = 1 / (q : ℚ) + 1 / (r : ℚ)) ↔ (p, q, r) ∈ SolutionSet :=
by sorry

end prime_equation_solutions_l2273_227363


namespace quadratic_inequality_solution_l2273_227353

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end quadratic_inequality_solution_l2273_227353


namespace quadratic_inequality_solution_l2273_227345

theorem quadratic_inequality_solution (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | x < a ∨ x > 1/a} := by sorry

end quadratic_inequality_solution_l2273_227345


namespace even_odd_sum_l2273_227313

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

/-- Given f and g are even and odd functions respectively, and f(x) - g(x) = x^3 + x^2 + 1,
    prove that f(1) + g(1) = 1 -/
theorem even_odd_sum (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g)
    (h : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 := by
  sorry

end even_odd_sum_l2273_227313


namespace min_value_a_min_value_a_achieved_l2273_227349

theorem min_value_a (a b : ℕ) (h1 : a > 0) (h2 : a = b - 2005) 
  (h3 : ∃ x : ℕ, x > 0 ∧ x^2 - a*x + b = 0) : a ≥ 95 := by
  sorry

theorem min_value_a_achieved (a b : ℕ) (h1 : a > 0) (h2 : a = b - 2005) : 
  (∃ x : ℕ, x > 0 ∧ x^2 - 95*x + (95 + 2005) = 0) := by
  sorry

end min_value_a_min_value_a_achieved_l2273_227349


namespace turning_process_terminates_l2273_227383

/-- Represents the direction a soldier is facing -/
inductive Direction
  | East
  | West

/-- Represents the state of the line of soldiers -/
def SoldierLine := List Direction

/-- Performs one step of the turning process -/
def turn_step (line : SoldierLine) : SoldierLine :=
  sorry

/-- Checks if the line is stable (no more turns needed) -/
def is_stable (line : SoldierLine) : Prop :=
  sorry

/-- The main theorem: the turning process will eventually stop -/
theorem turning_process_terminates (initial_line : SoldierLine) :
  ∃ (n : ℕ) (final_line : SoldierLine), 
    (n.iterate turn_step initial_line = final_line) ∧ is_stable final_line :=
  sorry

end turning_process_terminates_l2273_227383


namespace solve_clothing_problem_l2273_227314

def clothing_problem (total : ℕ) (num_loads : ℕ) (pieces_per_load : ℕ) : Prop :=
  let remaining := num_loads * pieces_per_load
  let first_load := total - remaining
  first_load = 19

theorem solve_clothing_problem :
  clothing_problem 39 5 4 := by
  sorry

end solve_clothing_problem_l2273_227314


namespace gcd_of_powers_of_three_l2273_227327

theorem gcd_of_powers_of_three : Nat.gcd (3^1200 - 1) (3^1210 - 1) = 3^10 - 1 := by sorry

end gcd_of_powers_of_three_l2273_227327


namespace max_dot_product_ellipses_l2273_227304

/-- The maximum dot product of vectors to points on two specific ellipses -/
theorem max_dot_product_ellipses : 
  let C₁ := {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 9 = 1}
  ∃ (max : ℝ), max = 15 ∧ 
    ∀ (M N : ℝ × ℝ), M ∈ C₁ → N ∈ C₂ → 
      (M.1 * N.1 + M.2 * N.2 : ℝ) ≤ max :=
by sorry

end max_dot_product_ellipses_l2273_227304


namespace new_boarders_correct_l2273_227321

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 15

/-- The initial number of boarders -/
def initial_boarders : ℕ := 60

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 2 / 5

/-- The final ratio of boarders to day students -/
def final_ratio : ℚ := 1 / 2

/-- The theorem stating that the number of new boarders is correct -/
theorem new_boarders_correct :
  let initial_day_students := (initial_boarders : ℚ) / initial_ratio
  (initial_boarders + new_boarders : ℚ) / initial_day_students = final_ratio :=
by sorry

end new_boarders_correct_l2273_227321


namespace squarefree_juicy_integers_l2273_227386

def is_juicy (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (d₁ d₂ : ℕ), d₁ ∣ n → d₂ ∣ n → d₁ < d₂ → (d₂ - d₁) ∣ n

def is_squarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p ∣ n → p = 1)

theorem squarefree_juicy_integers :
  {n : ℕ | is_squarefree n ∧ is_juicy n} = {2, 6, 42, 1806} :=
sorry

end squarefree_juicy_integers_l2273_227386


namespace perpendicular_line_equation_l2273_227394

/-- A line passing through (0, 7) perpendicular to 2x - 6y - 14 = 0 has equation y + 3x - 7 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), y = m * x + b ∧ (0, 7) ∈ {(x, y) | y = m * x + b}) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 2 * x₁ - 6 * y₁ - 14 = 0 ∧ 2 * x₂ - 6 * y₂ - 14 = 0 → 
    (y₂ - y₁) * (y - 7) = -(x₂ - x₁) * x) → 
  y + 3 * x - 7 = 0 := by
sorry

end perpendicular_line_equation_l2273_227394


namespace polyhedron_problem_l2273_227375

/-- Represents a convex polyhedron with hexagonal and quadrilateral faces. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  hexagons : ℕ
  quadrilaterals : ℕ
  H : ℕ
  Q : ℕ

/-- Euler's formula for convex polyhedra -/
def euler_formula (p : Polyhedron) : Prop :=
  p.vertices - p.edges + p.faces = 2

/-- The number of edges in terms of hexagons and quadrilaterals -/
def edge_count (p : Polyhedron) : Prop :=
  p.edges = 2 * p.quadrilaterals + 3 * p.hexagons

/-- Theorem about the specific polyhedron described in the problem -/
theorem polyhedron_problem :
  ∀ p : Polyhedron,
    p.faces = 44 →
    p.hexagons = 12 →
    p.quadrilaterals = 32 →
    p.H = 2 →
    p.Q = 2 →
    euler_formula p →
    edge_count p →
    100 * p.H + 10 * p.Q + p.vertices = 278 := by
  sorry

end polyhedron_problem_l2273_227375


namespace quadratic_equation_m_value_l2273_227333

theorem quadratic_equation_m_value : 
  ∀ m : ℝ, 
  (∀ x : ℝ, (m - 1) * x^(|m| + 1) + 2 * m * x + 3 = 0 → 
    (|m| + 1 = 2 ∧ m - 1 ≠ 0)) → 
  m = -1 := by
sorry

end quadratic_equation_m_value_l2273_227333


namespace total_distance_to_school_l2273_227324

-- Define the distances
def bus_distance_km : ℝ := 2
def walking_distance_m : ℝ := 560

-- Define the conversion factor
def km_to_m : ℝ := 1000

-- Theorem to prove
theorem total_distance_to_school :
  bus_distance_km * km_to_m + walking_distance_m = 2560 := by
  sorry

end total_distance_to_school_l2273_227324


namespace inscribed_circle_triangle_shortest_side_l2273_227367

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  s1 : ℝ
  /-- The length of the second segment of the divided side -/
  s2 : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ

/-- Theorem stating that for a triangle with an inscribed circle of radius 5 units
    that divides one side into segments of 4 and 10 units, the shortest side is 30 units -/
theorem inscribed_circle_triangle_shortest_side
  (t : InscribedCircleTriangle)
  (h1 : t.r = 5)
  (h2 : t.s1 = 4)
  (h3 : t.s2 = 10) :
  t.shortest_side = 30 := by
  sorry


end inscribed_circle_triangle_shortest_side_l2273_227367


namespace sum_of_squares_zero_implies_sum_l2273_227372

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end sum_of_squares_zero_implies_sum_l2273_227372


namespace book_price_percentage_l2273_227365

theorem book_price_percentage (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) : 
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.6 * marked_price
  alice_paid / suggested_retail_price = 0.36 := by
sorry

end book_price_percentage_l2273_227365


namespace elliptical_path_derivative_l2273_227397

/-- The derivative of a vector function representing an elliptical path. -/
theorem elliptical_path_derivative (a b t : ℝ) :
  let r : ℝ → ℝ × ℝ := fun t => (a * Real.cos t, b * Real.sin t)
  let dr : ℝ → ℝ × ℝ := fun t => (-a * Real.sin t, b * Real.cos t)
  (deriv r) t = dr t := by
  sorry

end elliptical_path_derivative_l2273_227397


namespace cube_sum_inequality_l2273_227326

theorem cube_sum_inequality (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) 
  (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 ∧ 
  (a + b + c + a^2 + b^2 + c^2 = 4 ↔ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = -1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = 1 ∧ c = -1) ∨ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = -1 ∧ c = 1)) :=
by sorry

end cube_sum_inequality_l2273_227326


namespace interior_angle_sum_difference_l2273_227310

/-- The sum of interior angles of a convex n-sided polygon in degrees -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

theorem interior_angle_sum_difference (n : ℕ) (h : n ≥ 3) :
  interior_angle_sum (n + 1) - interior_angle_sum n = 180 := by
  sorry

end interior_angle_sum_difference_l2273_227310


namespace quadratic_function_range_l2273_227342

/-- Given a quadratic function f(x) = ax^2 + bx, where 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4,
    the range of f(-2) is [6, 10]. -/
theorem quadratic_function_range (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) ∧ (2 ≤ f 1 ∧ f 1 ≤ 4) →
  6 ≤ f (-2) ∧ f (-2) ≤ 10 :=
by sorry

end quadratic_function_range_l2273_227342


namespace quadratic_root_property_l2273_227332

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 100 = 0 → a^4 - 201*a = 10100 := by
  sorry

end quadratic_root_property_l2273_227332


namespace square_of_good_is_good_l2273_227340

def is_averaging_sequence (a : ℕ → ℤ) : Prop :=
  ∀ k, 2 * a (k + 1) = a k + a (k + 1)

def is_good_sequence (x : ℕ → ℤ) : Prop :=
  ∀ n, is_averaging_sequence (λ k => x (n + k))

theorem square_of_good_is_good (x : ℕ → ℤ) :
  is_good_sequence x → is_good_sequence (λ k => x k ^ 2) := by
  sorry

end square_of_good_is_good_l2273_227340


namespace eggs_in_club_house_l2273_227374

theorem eggs_in_club_house (total eggs_in_park eggs_in_town_hall eggs_in_club_house : ℕ) :
  total = eggs_in_club_house + eggs_in_park + eggs_in_town_hall →
  eggs_in_park = 25 →
  eggs_in_town_hall = 15 →
  total = 80 →
  eggs_in_club_house = 40 := by
sorry

end eggs_in_club_house_l2273_227374


namespace rental_cost_equality_l2273_227361

/-- The daily rate for Sunshine Car Rentals in dollars -/
def sunshine_base : ℝ := 17.99

/-- The per-mile rate for Sunshine Car Rentals in dollars -/
def sunshine_per_mile : ℝ := 0.18

/-- The daily rate for City Rentals in dollars -/
def city_base : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_per_mile : ℝ := 0.16

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 48

theorem rental_cost_equality :
  sunshine_base + sunshine_per_mile * equal_cost_mileage =
  city_base + city_per_mile * equal_cost_mileage :=
by sorry

end rental_cost_equality_l2273_227361


namespace arithmetic_sequence_ratio_l2273_227347

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = (n : ℝ) * (a 0 + a (n-1)) / 2

/-- Theorem: If S_n / S_2n = (n+1) / (4n+2) for an arithmetic sequence,
    then a_3 / a_5 = 3/5 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : ∀ n, seq.S n / seq.S (2*n) = (n + 1 : ℝ) / (4*n + 2)) : 
  seq.a 3 / seq.a 5 = 3/5 := by
  sorry

end arithmetic_sequence_ratio_l2273_227347


namespace vector_problem_l2273_227364

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, 1)

theorem vector_problem :
  (∃ θ : ℝ, θ = Real.arccos ((-3 * 1 + 1 * (-2)) / (Real.sqrt 10 * Real.sqrt 5)) ∧ θ = 3 * π / 4) ∧
  (∃ k : ℝ, (∃ t : ℝ, t ≠ 0 ∧ c = t • (a + k • b)) ∧ k = 4 / 3) :=
by sorry

end vector_problem_l2273_227364


namespace intersection_implies_m_equals_two_sufficient_condition_implies_m_range_l2273_227385

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_equals_two :
  ∀ m : ℝ, A ∩ B m = Set.Icc 0 3 → m = 2 := by sorry

-- Theorem 2
theorem sufficient_condition_implies_m_range :
  ∀ m : ℝ, A ⊆ Set.univ \ B m → m > 5 ∨ m < -3 := by sorry

end intersection_implies_m_equals_two_sufficient_condition_implies_m_range_l2273_227385


namespace exists_n_in_sequence_l2273_227336

theorem exists_n_in_sequence (a : ℕ → ℕ) : (∀ n, a n = n^2 + n) → ∃ n, a n = 30 := by
  sorry

end exists_n_in_sequence_l2273_227336


namespace carrots_and_cauliflower_cost_l2273_227355

/-- The cost of a bunch of carrots and a cauliflower given specific pricing conditions -/
theorem carrots_and_cauliflower_cost :
  ∀ (p c f o : ℝ),
    p + c + f + o = 30 →  -- Total cost
    o = 3 * p →           -- Oranges cost thrice potatoes
    f = p + c →           -- Cauliflower costs sum of potatoes and carrots
    c + f = 14 :=
by
  sorry

end carrots_and_cauliflower_cost_l2273_227355


namespace fraction_equality_l2273_227357

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end fraction_equality_l2273_227357


namespace functional_equation_solution_l2273_227387

theorem functional_equation_solution (c : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)) →
  c = 1 ∨ c = -1 := by
sorry

end functional_equation_solution_l2273_227387


namespace horner_v3_value_l2273_227337

def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_v3_value :
  horner_v3 f (-4) = -57 := by sorry

end horner_v3_value_l2273_227337


namespace amelia_win_probability_l2273_227317

/-- Probability of Amelia's coin landing on heads -/
def p_amelia : ℚ := 3/7

/-- Probability of Blaine's coin landing on heads -/
def p_blaine : ℚ := 1/3

/-- Probability of at least one head in a simultaneous toss -/
def p_start : ℚ := 1 - (1 - p_amelia) * (1 - p_blaine)

/-- Probability of Amelia winning on her turn -/
def p_amelia_win : ℚ := p_amelia * p_amelia

/-- Probability of Blaine winning on his turn -/
def p_blaine_win : ℚ := p_blaine * p_blaine

/-- Probability of delay (neither wins) -/
def p_delay : ℚ := 1 - p_amelia_win - p_blaine_win

/-- The probability that Amelia wins the game -/
theorem amelia_win_probability : 
  (p_amelia_win / (1 - p_delay^2 : ℚ)) = 21609/64328 := by
  sorry

end amelia_win_probability_l2273_227317


namespace tobias_allowance_is_five_l2273_227399

/-- Represents Tobias's financial situation --/
structure TobiasFinances where
  shoe_cost : ℕ
  saving_months : ℕ
  change : ℕ
  lawns_mowed : ℕ
  lawn_price : ℕ
  driveways_shoveled : ℕ
  driveway_price : ℕ

/-- Calculates Tobias's monthly allowance --/
def monthly_allowance (tf : TobiasFinances) : ℕ :=
  let total_earned := tf.lawns_mowed * tf.lawn_price + tf.driveways_shoveled * tf.driveway_price
  let total_had := tf.shoe_cost + tf.change
  let allowance_total := total_had - total_earned
  allowance_total / tf.saving_months

/-- Theorem stating that Tobias's monthly allowance is $5 --/
theorem tobias_allowance_is_five (tf : TobiasFinances) 
  (h1 : tf.shoe_cost = 95)
  (h2 : tf.saving_months = 3)
  (h3 : tf.change = 15)
  (h4 : tf.lawns_mowed = 4)
  (h5 : tf.lawn_price = 15)
  (h6 : tf.driveways_shoveled = 5)
  (h7 : tf.driveway_price = 7) :
  monthly_allowance tf = 5 := by
  sorry

end tobias_allowance_is_five_l2273_227399


namespace altered_coin_probability_l2273_227338

theorem altered_coin_probability :
  ∃! p : ℝ, 0 < p ∧ p < 1/2 ∧ (20 : ℝ) * p^3 * (1-p)^3 = 1/8 := by
  sorry

end altered_coin_probability_l2273_227338


namespace max_value_of_f_l2273_227395

noncomputable def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem max_value_of_f :
  ∃ (M : ℝ), M = 23 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end max_value_of_f_l2273_227395


namespace complex_equation_solution_l2273_227379

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end complex_equation_solution_l2273_227379


namespace intersection_M_N_l2273_227396

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the open interval (0, 1]
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = open_unit_interval := by sorry

end intersection_M_N_l2273_227396


namespace man_running_opposite_direction_l2273_227380

/-- Proves that the relative speed between a train and a man is equal to the sum of their speeds,
    indicating that the man is running in the opposite direction to the train. -/
theorem man_running_opposite_direction
  (train_length : ℝ)
  (train_speed : ℝ)
  (man_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed = 40 * 1000 / 3600)
  (h3 : man_speed = 4 * 1000 / 3600)
  (h4 : passing_time = 9) :
  train_length / passing_time = train_speed + man_speed := by
  sorry

#check man_running_opposite_direction

end man_running_opposite_direction_l2273_227380


namespace mayas_books_pages_l2273_227306

/-- Proves that if Maya read 5 books last week, read twice as much this week, 
    and read a total of 4500 pages this week, then each book had 450 pages. -/
theorem mayas_books_pages (books_last_week : ℕ) (pages_this_week : ℕ) 
  (h1 : books_last_week = 5)
  (h2 : pages_this_week = 4500) :
  (pages_this_week / (2 * books_last_week) : ℚ) = 450 := by
  sorry

#check mayas_books_pages

end mayas_books_pages_l2273_227306


namespace chemistry_physics_difference_l2273_227331

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℕ
  physics : ℕ
  chemistry : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.math + s.physics = 70 ∧
  s.chemistry > s.physics ∧
  (s.math + s.chemistry) / 2 = 45

/-- The theorem to be proved -/
theorem chemistry_physics_difference (s : Scores) 
  (h : satisfiesConditions s) : s.chemistry - s.physics = 20 := by
  sorry

end chemistry_physics_difference_l2273_227331


namespace carson_gardening_time_l2273_227371

/-- Represents the gardening tasks Carson needs to complete -/
structure GardeningTasks where
  mow_lines : ℕ
  mow_time_per_line : ℕ
  flower_rows : ℕ
  flowers_per_row : ℕ
  planting_time_per_flower : ℚ
  garden_sections : ℕ
  watering_time_per_section : ℕ
  hedges : ℕ
  trimming_time_per_hedge : ℕ

/-- Calculates the total gardening time in minutes -/
def total_gardening_time (tasks : GardeningTasks) : ℚ :=
  tasks.mow_lines * tasks.mow_time_per_line +
  tasks.flower_rows * tasks.flowers_per_row * tasks.planting_time_per_flower +
  tasks.garden_sections * tasks.watering_time_per_section +
  tasks.hedges * tasks.trimming_time_per_hedge

/-- Theorem stating that Carson's total gardening time is 162 minutes -/
theorem carson_gardening_time :
  let tasks : GardeningTasks := {
    mow_lines := 40,
    mow_time_per_line := 2,
    flower_rows := 10,
    flowers_per_row := 8,
    planting_time_per_flower := 1/2,
    garden_sections := 4,
    watering_time_per_section := 3,
    hedges := 5,
    trimming_time_per_hedge := 6
  }
  total_gardening_time tasks = 162 := by
  sorry


end carson_gardening_time_l2273_227371


namespace sin_negative_nineteen_pi_sixths_l2273_227382

theorem sin_negative_nineteen_pi_sixths (π : Real) : 
  let sine_is_odd : ∀ x, Real.sin (-x) = -Real.sin x := by sorry
  let sine_period : ∀ x, Real.sin (x + 2 * π) = Real.sin x := by sorry
  let sine_cofunction : ∀ θ, Real.sin (π + θ) = -Real.sin θ := by sorry
  let sin_pi_sixth : Real.sin (π / 6) = 1 / 2 := by sorry
  Real.sin (-19 * π / 6) = 1 / 2 := by sorry

end sin_negative_nineteen_pi_sixths_l2273_227382


namespace quadratic_reciprocal_roots_l2273_227302

theorem quadratic_reciprocal_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x * y = 1 ∧ x^2 - 2*(m+2)*x + m^2 - 4 = 0 ∧ y^2 - 2*(m+2)*y + m^2 - 4 = 0) 
  → m = Real.sqrt 5 := by
  sorry

end quadratic_reciprocal_roots_l2273_227302


namespace compound_interest_period_l2273_227323

theorem compound_interest_period (P s k n : ℝ) (h_pos : k > -1) :
  P = s / (1 + k)^n →
  n = Real.log (s/P) / Real.log (1 + k) :=
by sorry

end compound_interest_period_l2273_227323


namespace frankies_pets_l2273_227377

/-- The number of pets Frankie has -/
def total_pets (cats : ℕ) : ℕ :=
  let snakes := 2 * cats
  let parrots := cats - 1
  let tortoises := parrots + 1
  let dogs := 2
  let hamsters := 3
  let fish := 5
  cats + snakes + parrots + tortoises + dogs + hamsters + fish

/-- Theorem stating the total number of Frankie's pets -/
theorem frankies_pets :
  ∃ (cats : ℕ),
    2 * cats + cats + 2 = 14 ∧
    total_pets cats = 39 := by
  sorry

end frankies_pets_l2273_227377


namespace remainder_of_111222333_div_37_l2273_227343

theorem remainder_of_111222333_div_37 : 111222333 % 37 = 0 := by
  sorry

end remainder_of_111222333_div_37_l2273_227343


namespace wheel_revolutions_for_one_mile_l2273_227350

-- Define the wheel diameter in feet
def wheel_diameter : ℝ := 8

-- Define one mile in feet
def mile_in_feet : ℝ := 5280

-- Theorem statement
theorem wheel_revolutions_for_one_mile :
  (mile_in_feet / (π * wheel_diameter)) = 660 / π := by
  sorry

end wheel_revolutions_for_one_mile_l2273_227350


namespace fraction_existence_and_nonexistence_l2273_227370

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ a / b ∧ a / b ≤ Real.sqrt (n + 1)) ∧
  (∃ f : ℕ+ → ℕ+, Function.Injective f ∧ ∀ n : ℕ+, ¬∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt (f n) ∧ Real.sqrt (f n) ≤ a / b ∧ a / b ≤ Real.sqrt (f n + 1)) :=
by
  sorry

end fraction_existence_and_nonexistence_l2273_227370


namespace triangle_area_l2273_227300

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 4 under the given conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = 2 → c = 5 → Real.cos B = 3/5 → 
  (1/2) * a * c * Real.sin B = 4 := by sorry

end triangle_area_l2273_227300


namespace sqrt_equation_solution_l2273_227335

theorem sqrt_equation_solution : ∃! z : ℚ, Real.sqrt (10 + 3 * z) = 12 := by
  sorry

end sqrt_equation_solution_l2273_227335


namespace largest_area_error_l2273_227319

theorem largest_area_error (actual_side : ℝ) (max_error_percent : ℝ) :
  actual_side = 30 →
  max_error_percent = 20 →
  let max_measured_side := actual_side * (1 + max_error_percent / 100)
  let actual_area := actual_side ^ 2
  let max_measured_area := max_measured_side ^ 2
  let max_percent_error := (max_measured_area - actual_area) / actual_area * 100
  max_percent_error = 44 := by
sorry

end largest_area_error_l2273_227319


namespace students_absent_eq_three_l2273_227354

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (2 * dozen) + (dozen / 2)

/-- The total number of people in the class (including Dani) -/
def total_people : ℕ := 29

/-- The number of cupcakes left after distribution -/
def cupcakes_left : ℕ := 4

/-- The number of students who called in sick -/
def students_absent : ℕ := total_people - (cupcakes_brought - cupcakes_left)

theorem students_absent_eq_three : students_absent = 3 := by
  sorry

end students_absent_eq_three_l2273_227354


namespace anthony_pencil_count_l2273_227358

/-- Given Anthony's initial pencil count and the number of pencils Kathryn gives him,
    prove that the total number of pencils Anthony has is equal to the sum of these two quantities. -/
theorem anthony_pencil_count (initial : ℕ) (given : ℕ) : initial + given = initial + given :=
by sorry

end anthony_pencil_count_l2273_227358


namespace alice_least_money_l2273_227315

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Bob : Person
  | Charlie : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℝ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q

axiom charlie_most : ∀ (p : Person), p ≠ Person.Charlie → money p < money Person.Charlie

axiom bob_more_than_alice : money Person.Alice < money Person.Bob

axiom dana_more_than_alice : money Person.Alice < money Person.Dana

axiom eve_between_alice_and_bob : 
  money Person.Alice < money Person.Eve ∧ money Person.Eve < money Person.Bob

-- State the theorem
theorem alice_least_money : 
  ∀ (p : Person), p ≠ Person.Alice → money Person.Alice < money p := by
  sorry

end alice_least_money_l2273_227315


namespace impossibleOneLight_l2273_227329

/- Define the grid size -/
def gridSize : Nat := 8

/- Define the state of the grid as a function from coordinates to bool -/
def GridState := Fin gridSize → Fin gridSize → Bool

/- Define the initial state where all bulbs are on -/
def initialState : GridState := fun _ _ => true

/- Define the toggle operation for a row -/
def toggleRow (state : GridState) (row : Fin gridSize) : GridState :=
  fun i j => if i = row then !state i j else state i j

/- Define the toggle operation for a column -/
def toggleColumn (state : GridState) (col : Fin gridSize) : GridState :=
  fun i j => if j = col then !state i j else state i j

/- Define a property that checks if exactly one bulb is on -/
def exactlyOneBulbOn (state : GridState) : Prop :=
  ∃! i j, state i j = true

/- The main theorem -/
theorem impossibleOneLight : 
  ¬∃ (toggleSequence : List (Bool × Fin gridSize)), 
    let finalState := toggleSequence.foldl 
      (fun acc (toggle) => 
        match toggle with
        | (true, n) => toggleRow acc n
        | (false, n) => toggleColumn acc n) 
      initialState
    exactlyOneBulbOn finalState :=
by
  sorry

end impossibleOneLight_l2273_227329


namespace single_stuffed_animal_cost_l2273_227309

/-- Represents the cost of items at a garage sale. -/
structure GarageSaleCost where
  magnet : ℝ
  sticker : ℝ
  stuffed_animals : ℝ
  toy_car : ℝ
  discount_rate : ℝ
  max_budget : ℝ

/-- Conditions for the garage sale problem. -/
def garage_sale_conditions (cost : GarageSaleCost) : Prop :=
  cost.magnet = 6 ∧
  cost.magnet = 3 * cost.sticker ∧
  cost.magnet = cost.stuffed_animals / 4 ∧
  cost.toy_car = cost.stuffed_animals / 4 ∧
  cost.toy_car = 2 * cost.sticker ∧
  cost.discount_rate = 0.1 ∧
  cost.max_budget = 30

/-- The theorem to be proved. -/
theorem single_stuffed_animal_cost 
  (cost : GarageSaleCost) 
  (h : garage_sale_conditions cost) : 
  cost.stuffed_animals / 2 = 12 := by
  sorry

end single_stuffed_animal_cost_l2273_227309


namespace jelly_bean_distribution_l2273_227330

/-- Represents the number of jelly beans each person has -/
structure JellyBeans :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Performs the first distribution: A gives to B and C -/
def firstDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a - jb.b - jb.c, jb.b + jb.b, jb.c + jb.c⟩

/-- Performs the second distribution: B gives to A and C -/
def secondDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a + jb.a, jb.b - jb.a - jb.c, jb.c + jb.c⟩

/-- Performs the third distribution: C gives to A and B -/
def thirdDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a + jb.a, jb.b + jb.b, jb.c - jb.a - jb.b⟩

theorem jelly_bean_distribution :
  let initial := JellyBeans.mk 104 56 32
  let final := thirdDistribution (secondDistribution (firstDistribution initial))
  final.a = 64 ∧ final.b = 64 ∧ final.c = 64 := by
  sorry

end jelly_bean_distribution_l2273_227330
