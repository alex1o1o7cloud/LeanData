import Mathlib

namespace sector_area_l2373_237335

/-- Given a circular sector with circumference 6 and central angle 1 radian, its area is 2 -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) :
  circumference = 6 →
  central_angle = 1 →
  area = 2 :=
by sorry

end sector_area_l2373_237335


namespace temperature_drop_l2373_237385

/-- Given an initial temperature and a temperature drop, calculates the final temperature -/
def finalTemperature (initial : Int) (drop : Int) : Int :=
  initial - drop

/-- Theorem: If the initial temperature is -6°C and it drops by 5°C, then the final temperature is -11°C -/
theorem temperature_drop : finalTemperature (-6) 5 = -11 := by
  sorry

end temperature_drop_l2373_237385


namespace polynomial_composition_l2373_237367

/-- Given a function f and a polynomial g, proves that g satisfies the given condition -/
theorem polynomial_composition (f g : ℝ → ℝ) : 
  (∀ x, f x = x^2) →
  (∀ x, f (g x) = 4*x^2 + 4*x + 1) →
  (∀ x, g x = 2*x + 1 ∨ g x = -2*x - 1) :=
by sorry

end polynomial_composition_l2373_237367


namespace arithmetic_sequence_fifth_term_l2373_237357

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_1_2 : a 1 + a 2 = 1)
  (h_sum_3_4 : a 3 + a 4 = 5) :
  a 5 = 4 :=
sorry

end arithmetic_sequence_fifth_term_l2373_237357


namespace partnership_profit_l2373_237396

/-- The total profit of a business partnership --/
def total_profit (a_investment b_investment : ℤ) (management_fee_percent : ℚ) (a_total_received : ℤ) : ℚ :=
  let total_investment := a_investment + b_investment
  let a_share_percent := a_investment / total_investment
  let remaining_profit_percent := 1 - management_fee_percent
  let a_total_percent := management_fee_percent + (a_share_percent * remaining_profit_percent)
  (a_total_received : ℚ) / a_total_percent

/-- The proposition that the total profit is 9600 given the specified conditions --/
theorem partnership_profit : 
  total_profit 15000 25000 (1/10) 4200 = 9600 := by
  sorry

end partnership_profit_l2373_237396


namespace distance_calculation_l2373_237325

def speed : Real := 20
def time : Real := 8
def distance : Real := speed * time

theorem distance_calculation : distance = 160 := by
  sorry

end distance_calculation_l2373_237325


namespace division_with_remainder_l2373_237363

theorem division_with_remainder (x y : ℕ+) : 
  (x : ℝ) / (y : ℝ) = 96.12 →
  (x : ℝ) % (y : ℝ) = 5.76 →
  y = 100 := by
sorry

end division_with_remainder_l2373_237363


namespace complex_equality_condition_l2373_237377

theorem complex_equality_condition (a b c d : ℝ) : 
  let z1 : ℂ := Complex.mk a b
  let z2 : ℂ := Complex.mk c d
  (z1 = z2 → a = c) ∧ 
  ∃ a b c d : ℝ, a = c ∧ Complex.mk a b ≠ Complex.mk c d :=
by sorry

end complex_equality_condition_l2373_237377


namespace max_students_distribution_l2373_237313

def stationery_A : ℕ := 38
def stationery_B : ℕ := 78
def stationery_C : ℕ := 128

def remaining_A : ℕ := 2
def remaining_B : ℕ := 6
def remaining_C : ℕ := 20

def distributed_A : ℕ := stationery_A - remaining_A
def distributed_B : ℕ := stationery_B - remaining_B
def distributed_C : ℕ := stationery_C - remaining_C

theorem max_students_distribution :
  ∃ (n : ℕ), n > 0 ∧ 
    distributed_A % n = 0 ∧
    distributed_B % n = 0 ∧
    distributed_C % n = 0 ∧
    ∀ (m : ℕ), m > n →
      (distributed_A % m ≠ 0 ∨
       distributed_B % m ≠ 0 ∨
       distributed_C % m ≠ 0) →
    n = 36 :=
by sorry

end max_students_distribution_l2373_237313


namespace roots_property_l2373_237344

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 + 5 * x - 7 = 0

-- Define the theorem
theorem roots_property (p q : ℝ) (hp : quadratic_eq p) (hq : quadratic_eq q) :
  (p - 2) * (q - 2) = 5 := by
  sorry

end roots_property_l2373_237344


namespace functional_equation_solution_l2373_237372

/-- Given functions f, g, h: ℝ → ℝ satisfying the functional equation
    f(x) - g(y) = (x-y) · h(x+y) for all x, y ∈ ℝ,
    prove that there exist constants d, c ∈ ℝ such that
    f(x) = g(x) = dx² + c for all x ∈ ℝ. -/
theorem functional_equation_solution
  (f g h : ℝ → ℝ)
  (h_eq : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, ∀ x : ℝ, f x = d * x^2 + c ∧ g x = d * x^2 + c :=
sorry

end functional_equation_solution_l2373_237372


namespace symmetric_points_difference_l2373_237342

/-- Given two points P₁ and P₂ that are symmetric with respect to the origin,
    prove that m - n = 8. -/
theorem symmetric_points_difference (m n : ℝ) : 
  (∃ (P₁ P₂ : ℝ × ℝ), 
    P₁ = (2 - m, 5) ∧ 
    P₂ = (3, 2*n + 1) ∧ 
    P₁.1 = -P₂.1 ∧ 
    P₁.2 = -P₂.2) → 
  m - n = 8 := by
sorry

end symmetric_points_difference_l2373_237342


namespace position_determination_in_plane_l2373_237351

theorem position_determination_in_plane :
  ∀ (P : ℝ × ℝ), ∃! (θ : ℝ) (r : ℝ), 
    P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ ∧ r ≥ 0 :=
by sorry

end position_determination_in_plane_l2373_237351


namespace heart_then_face_prob_l2373_237328

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suit of a card -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- The rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Definition of a face card -/
def isFaceCard (c : Card) : Prop :=
  c.rank = Rank.Jack ∨ c.rank = Rank.Queen ∨ c.rank = Rank.King ∨ c.rank = Rank.Ace

/-- The probability of drawing a heart as the first card and a face card as the second -/
def heartThenFaceProbability (d : Deck) : ℚ :=
  5 / 86

/-- Theorem stating the probability of drawing a heart then a face card -/
theorem heart_then_face_prob (d : Deck) :
  heartThenFaceProbability d = 5 / 86 := by
  sorry


end heart_then_face_prob_l2373_237328


namespace first_system_solution_second_system_solution_l2373_237369

-- First system of equations
theorem first_system_solution :
  ∃ (x y : ℝ), 3 * x + 2 * y = 5 ∧ y = 2 * x - 8 ∧ x = 3 ∧ y = -2 := by
sorry

-- Second system of equations
theorem second_system_solution :
  ∃ (x y : ℝ), 2 * x - y = 10 ∧ 2 * x + 3 * y = 2 ∧ x = 4 ∧ y = -2 := by
sorry

end first_system_solution_second_system_solution_l2373_237369


namespace complex_number_in_first_quadrant_l2373_237382

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - I) / (1 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l2373_237382


namespace inequality_proof_l2373_237371

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a + 2 * a * b * c = 1) :
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ 3 / 2 := by
  sorry

end inequality_proof_l2373_237371


namespace equation_solution_l2373_237352

theorem equation_solution : ∃! x : ℝ, 
  Real.sqrt x + Real.sqrt (x + 9) + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (3*x + 27) = 45 - 3*x ∧ 
  x = 729/144 := by
  sorry

end equation_solution_l2373_237352


namespace gcd_powers_of_two_l2373_237346

theorem gcd_powers_of_two : Nat.gcd (2^2024 - 1) (2^2007 - 1) = 2^17 - 1 := by
  sorry

end gcd_powers_of_two_l2373_237346


namespace abc_inequalities_l2373_237397

theorem abc_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
  ((1 + a) * (1 + b) * (1 + c) ≥ 8) ∧ 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 1/a + 1/b + 1/c) := by
  sorry

end abc_inequalities_l2373_237397


namespace a_seven_minus_a_two_l2373_237307

def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem a_seven_minus_a_two : a 7 - a 2 = 20 := by
  sorry

end a_seven_minus_a_two_l2373_237307


namespace shirt_original_price_l2373_237310

/-- Calculates the original price of an item given its discounted price and discount percentage. -/
def originalPrice (discountedPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  discountedPrice / (1 - discountPercentage / 100)

/-- Theorem stating that if a shirt is sold at Rs. 780 after a 20% discount, 
    then the original price of the shirt was Rs. 975. -/
theorem shirt_original_price : 
  originalPrice 780 20 = 975 := by
  sorry

end shirt_original_price_l2373_237310


namespace hyperbola_sufficient_not_necessary_l2373_237338

/-- Hyperbola equation -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Asymptotes equation -/
def is_asymptote (x y a b : ℝ) : Prop :=
  y = b/a * x ∨ y = -b/a * x

/-- The hyperbola equation is a sufficient but not necessary condition for the asymptotes equation -/
theorem hyperbola_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, is_hyperbola x y a b → is_asymptote x y a b) ∧
  ¬(∀ x y, is_asymptote x y a b → is_hyperbola x y a b) :=
sorry

end hyperbola_sufficient_not_necessary_l2373_237338


namespace min_stable_stories_l2373_237321

/-- Represents a domino placement on a rectangular grid --/
structure DominoPlacement :=
  (width : Nat) -- Width of the rectangle
  (height : Nat) -- Height of the rectangle
  (dominoes : Nat) -- Number of dominoes per story

/-- Represents a tower of domino placements --/
structure DominoTower :=
  (base : DominoPlacement)
  (stories : Nat)

/-- Defines when a domino tower is considered stable --/
def isStable (tower : DominoTower) : Prop :=
  ∀ (x y : ℚ), 0 ≤ x ∧ x < tower.base.width ∧ 0 ≤ y ∧ y < tower.base.height →
    ∃ (s : Nat), s < tower.stories ∧ 
      ∃ (dx dy : ℚ), (0 ≤ dx ∧ dx < 2 ∧ 0 ≤ dy ∧ dy < 1) ∧
        (⌊x⌋ ≤ x - dx ∧ x - dx < ⌊x⌋ + 1) ∧
        (⌊y⌋ ≤ y - dy ∧ y - dy < ⌊y⌋ + 1)

/-- The main theorem stating the minimum number of stories for a stable tower --/
theorem min_stable_stories (tower : DominoTower) 
  (h_width : tower.base.width = 10)
  (h_height : tower.base.height = 11)
  (h_dominoes : tower.base.dominoes = 55) :
  (isStable tower ↔ tower.stories ≥ 5) :=
sorry

end min_stable_stories_l2373_237321


namespace ten_students_both_activities_l2373_237361

/-- Calculates the number of students who can do both swimming and gymnastics -/
def students_both_activities (total : ℕ) (swim : ℕ) (gym : ℕ) (neither : ℕ) : ℕ :=
  total - (total - swim + total - gym - neither)

/-- Theorem stating that 10 students can do both swimming and gymnastics -/
theorem ten_students_both_activities :
  students_both_activities 60 27 28 15 = 10 := by
  sorry

end ten_students_both_activities_l2373_237361


namespace remainder_of_12345678_div_9_l2373_237376

theorem remainder_of_12345678_div_9 : 12345678 % 9 = 0 := by
  sorry

end remainder_of_12345678_div_9_l2373_237376


namespace test_composition_l2373_237381

theorem test_composition (total_points total_questions : ℕ) 
  (h1 : total_points = 100) 
  (h2 : total_questions = 40) : 
  ∃ (two_point_questions four_point_questions : ℕ),
    two_point_questions + four_point_questions = total_questions ∧
    2 * two_point_questions + 4 * four_point_questions = total_points ∧
    two_point_questions = 30 := by
  sorry

end test_composition_l2373_237381


namespace cyclic_sum_inequality_l2373_237306

open Real

/-- The cyclic sum of a function over five variables -/
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ) (a b c d e : ℝ) : ℝ :=
  f a b c d e + f b c d e a + f c d e a b + f d e a b c + f e a b c d

/-- Theorem: For positive real numbers a, b, c, d, e satisfying abcde = 1,
    the cyclic sum of (a + abc)/(1 + ab + abcd) is greater than or equal to 10/3 -/
theorem cyclic_sum_inequality (a b c d e : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
    (h_prod : a * b * c * d * e = 1) :
    cyclicSum (fun a b c d e => (a + a*b*c)/(1 + a*b + a*b*c*d)) a b c d e ≥ 10/3 := by
  sorry

end cyclic_sum_inequality_l2373_237306


namespace equation_solution_l2373_237308

theorem equation_solution : ∃ (x : ℝ), (3 / (x - 2) - 1 = 1 / (2 - x)) ∧ (x = 6) := by
  sorry

end equation_solution_l2373_237308


namespace prime_sum_theorem_l2373_237370

theorem prime_sum_theorem (a b c : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → 
  b + c = 13 → c^2 - a^2 = 72 → 
  a + b + c = 20 := by
  sorry

end prime_sum_theorem_l2373_237370


namespace hike_duration_is_one_hour_l2373_237316

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  total_distance : Real
  initial_water : Real
  final_water : Real
  leak_rate : Real
  last_mile_consumption : Real
  first_three_miles_rate : Real

/-- Calculates the duration of the hike based on given conditions -/
def hike_duration (scenario : HikeScenario) : Real :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the hike duration is 1 hour for the given scenario -/
theorem hike_duration_is_one_hour (scenario : HikeScenario) 
  (h1 : scenario.total_distance = 4)
  (h2 : scenario.initial_water = 6)
  (h3 : scenario.final_water = 1)
  (h4 : scenario.leak_rate = 1)
  (h5 : scenario.last_mile_consumption = 1)
  (h6 : scenario.first_three_miles_rate = 0.6666666666666666) :
  hike_duration scenario = 1 := by
  sorry

end hike_duration_is_one_hour_l2373_237316


namespace shane_photos_february_l2373_237364

/-- The number of photos Shane takes in the first two months of the year -/
def total_photos : ℕ := 146

/-- The number of photos Shane takes each day in January -/
def photos_per_day_january : ℕ := 2

/-- The number of days in January -/
def days_in_january : ℕ := 31

/-- The number of weeks in February -/
def weeks_in_february : ℕ := 4

/-- Calculate the number of photos Shane takes each week in February -/
def photos_per_week_february : ℕ :=
  (total_photos - photos_per_day_january * days_in_january) / weeks_in_february

theorem shane_photos_february :
  photos_per_week_february = 21 := by
  sorry

end shane_photos_february_l2373_237364


namespace triangle_cosine_theorem_l2373_237347

/-- Given a triangle ABC with sides BC = 5 and AC = 4, and cos(A - B) = 7/8, prove that cos C = -1/4 -/
theorem triangle_cosine_theorem (A B C : ℝ) (h1 : BC = 5) (h2 : AC = 4) (h3 : Real.cos (A - B) = 7/8) :
  Real.cos C = -1/4 := by
  sorry

end triangle_cosine_theorem_l2373_237347


namespace base7_subtraction_l2373_237320

/-- Converts a base-7 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a decimal number to base-7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem base7_subtraction :
  let a := [5, 5, 2, 1]  -- 1255 in base 7
  let b := [2, 3, 4]     -- 432 in base 7
  let c := [1, 2, 5]     -- 521 in base 7
  toBase7 (toDecimal a - toDecimal b) = c := by sorry

end base7_subtraction_l2373_237320


namespace circle_diameter_endpoint_l2373_237350

/-- Given a circle with center (1, -2) and one endpoint of a diameter at (4, 3),
    the other endpoint of the diameter is at (7, 3). -/
theorem circle_diameter_endpoint :
  let center : ℝ × ℝ := (1, -2)
  let endpoint1 : ℝ × ℝ := (4, 3)
  let endpoint2 : ℝ × ℝ := (7, 3)
  (endpoint1.1 - center.1 = center.1 - endpoint2.1 ∧
   endpoint1.2 - center.2 = center.2 - endpoint2.2) :=
by sorry

end circle_diameter_endpoint_l2373_237350


namespace sod_area_second_section_l2373_237387

/-- Given the total area of sod needed and the area of the first section,
    prove that the area of the second section is 4800 square feet. -/
theorem sod_area_second_section
  (total_sod_squares : ℕ)
  (sod_square_size : ℕ)
  (first_section_length : ℕ)
  (first_section_width : ℕ)
  (h1 : total_sod_squares = 1500)
  (h2 : sod_square_size = 4)
  (h3 : first_section_length = 30)
  (h4 : first_section_width = 40) :
  total_sod_squares * sod_square_size - first_section_length * first_section_width = 4800 :=
by sorry

end sod_area_second_section_l2373_237387


namespace g_of_two_l2373_237300

/-- Given a function g: ℝ → ℝ that satisfies g(x) - 2g(1/x) = 3^x for all x ≠ 0,
    prove that g(2) = -3 - (4√3)/9 -/
theorem g_of_two (g : ℝ → ℝ) (h : ∀ x ≠ 0, g x - 2 * g (1/x) = 3^x) :
  g 2 = -3 - (4 * Real.sqrt 3) / 9 := by
  sorry

end g_of_two_l2373_237300


namespace unique_prime_triple_l2373_237360

theorem unique_prime_triple (p : ℤ) : 
  (Nat.Prime p.natAbs ∧ Nat.Prime (p + 2).natAbs ∧ Nat.Prime (p + 4).natAbs) ↔ p = 3 :=
sorry

end unique_prime_triple_l2373_237360


namespace range_of_2x_plus_y_range_of_c_l2373_237348

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Statement for the range of 2x + y
theorem range_of_2x_plus_y (x y : ℝ) (h : Circle x y) :
  -1 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 5 := by sorry

-- Statement for the range of c
theorem range_of_c (c : ℝ) (h : ∀ x y : ℝ, Circle x y → x + y + c > 0) :
  c ≥ -1 := by sorry

end range_of_2x_plus_y_range_of_c_l2373_237348


namespace grape_pickers_l2373_237329

/-- Given information about grape pickers and their work rate, calculate the number of pickers. -/
theorem grape_pickers (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ) :
  total_drums = 90 →
  total_days = 6 →
  drums_per_day = 15 →
  (total_drums / total_days : ℚ) = drums_per_day →
  drums_per_day / drums_per_day = 1 :=
by sorry

end grape_pickers_l2373_237329


namespace invisible_dots_count_l2373_237304

/-- The sum of numbers on a single six-sided die -/
def die_sum : Nat := 21

/-- The total number of dots on four dice -/
def total_dots : Nat := 4 * die_sum

/-- The sum of visible numbers on the dice -/
def visible_sum : Nat := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6

/-- The number of dots not visible on the dice -/
def invisible_dots : Nat := total_dots - visible_sum

theorem invisible_dots_count : invisible_dots = 54 := by
  sorry

end invisible_dots_count_l2373_237304


namespace line_through_origin_and_intersection_l2373_237319

-- Define the two lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y + 8 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (x, y) where
  x := -1
  y := -2

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Theorem statement
theorem line_through_origin_and_intersection :
  ∃ (x y : ℝ),
    line1 x y ∧ 
    line2 x y ∧ 
    line_l 0 0 ∧ 
    line_l (intersection_point.1) (intersection_point.2) ∧
    ∀ (a b : ℝ), line_l a b ↔ 2*a - b = 0 :=
sorry

end line_through_origin_and_intersection_l2373_237319


namespace problem_statement_l2373_237359

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is a given number of standard deviations below the mean -/
def value_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- The problem statement -/
theorem problem_statement (d : NormalDistribution) 
  (h1 : d.mean = 17.5)
  (h2 : d.std_dev = 2.5) :
  value_below_mean d 2.7 = 10.75 := by
  sorry

end problem_statement_l2373_237359


namespace power_factorial_inequality_l2373_237303

theorem power_factorial_inequality (n : ℕ) : 2^n * n.factorial < (n + 1)^n := by
  sorry

end power_factorial_inequality_l2373_237303


namespace rabbit_chicken_problem_l2373_237324

theorem rabbit_chicken_problem (total : ℕ) (rabbits chickens : ℕ → ℕ) :
  total = 40 →
  (∀ x : ℕ, rabbits x + chickens x = total) →
  (∀ x : ℕ, 4 * rabbits x = 10 * 2 * chickens x - 8) →
  (∃ x : ℕ, rabbits x = 33) :=
by sorry

end rabbit_chicken_problem_l2373_237324


namespace symmetry_of_exponential_graphs_l2373_237340

theorem symmetry_of_exponential_graphs :
  ∀ a : ℝ, 
  let f : ℝ → ℝ := λ x => 3^x
  let g : ℝ → ℝ := λ x => -(3^(-x))
  (f a = 3^a ∧ g (-a) = -3^a) ∧ 
  ((-a, -f a) = (-1 : ℝ) • (a, f a)) := by sorry

end symmetry_of_exponential_graphs_l2373_237340


namespace final_pet_count_l2373_237354

/-- Represents the number of pets in the pet center -/
structure PetCount where
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  birds : ℕ

/-- Calculates the total number of pets -/
def totalPets (pets : PetCount) : ℕ :=
  pets.dogs + pets.cats + pets.rabbits + pets.birds

/-- Initial pet count -/
def initialPets : PetCount :=
  { dogs := 36, cats := 29, rabbits := 15, birds := 10 }

/-- First adoption -/
def firstAdoption (pets : PetCount) : PetCount :=
  { dogs := pets.dogs - 20, cats := pets.cats, rabbits := pets.rabbits - 5, birds := pets.birds }

/-- New pets added -/
def newPetsAdded (pets : PetCount) : PetCount :=
  { dogs := pets.dogs, cats := pets.cats + 12, rabbits := pets.rabbits + 8, birds := pets.birds + 5 }

/-- Second adoption -/
def secondAdoption (pets : PetCount) : PetCount :=
  { dogs := pets.dogs, cats := pets.cats - 10, rabbits := pets.rabbits, birds := pets.birds - 4 }

/-- The main theorem stating the final number of pets -/
theorem final_pet_count :
  totalPets (secondAdoption (newPetsAdded (firstAdoption initialPets))) = 76 := by
  sorry

end final_pet_count_l2373_237354


namespace diamond_five_three_l2373_237368

-- Define the operation ⋄
def diamond (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem statement
theorem diamond_five_three : diamond 5 3 = 38 := by
  sorry

end diamond_five_three_l2373_237368


namespace locus_of_centers_l2373_237391

-- Define the circles C₃ and C₄
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₄ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being externally tangent to C₃ and internally tangent to C₄
def is_tangent_to_C₃_C₄ (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)

-- State the theorem
theorem locus_of_centers :
  ∀ a b : ℝ, (∃ r : ℝ, is_tangent_to_C₃_C₄ a b r) → a^2 + 18*b^2 - 6*a - 440 = 0 :=
sorry

end locus_of_centers_l2373_237391


namespace tims_takeout_cost_l2373_237395

/-- The total cost of Tim's Chinese take-out -/
def total_cost : ℝ := 50

/-- The percentage of the cost that went to entrees -/
def entree_percentage : ℝ := 0.8

/-- The number of appetizers Tim bought -/
def num_appetizers : ℕ := 2

/-- The cost of a single appetizer -/
def appetizer_cost : ℝ := 5

theorem tims_takeout_cost :
  total_cost = (num_appetizers : ℝ) * appetizer_cost / (1 - entree_percentage) :=
by sorry

end tims_takeout_cost_l2373_237395


namespace sufficient_not_necessary_condition_l2373_237388

/-- The equation of a potential hyperbola with parameter k -/
def hyperbola_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k - 3) - y^2 / (k + 3) = 1

/-- Predicate to check if an equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y, hyperbola_equation k x y ∧ (k - 3) * (k + 3) > 0

/-- Statement: k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_not_necessary_condition :
  (∀ k : ℝ, k > 3 → is_hyperbola k) ∧
  ¬(∀ k : ℝ, is_hyperbola k → k > 3) :=
sorry

end sufficient_not_necessary_condition_l2373_237388


namespace power_division_equality_l2373_237366

theorem power_division_equality : (3 : ℕ)^16 / (81 : ℕ)^2 = 6561 := by
  sorry

end power_division_equality_l2373_237366


namespace chef_guests_problem_l2373_237345

theorem chef_guests_problem (adults children seniors : ℕ) : 
  children = adults - 35 →
  seniors = 2 * children →
  adults + children + seniors = 127 →
  adults = 58 := by
  sorry

end chef_guests_problem_l2373_237345


namespace total_toys_l2373_237365

theorem total_toys (jaxon_toys gabriel_toys jerry_toys : ℕ) : 
  jaxon_toys = 15 →
  gabriel_toys = 2 * jaxon_toys →
  jerry_toys = gabriel_toys + 8 →
  jaxon_toys + gabriel_toys + jerry_toys = 83 := by
sorry

end total_toys_l2373_237365


namespace percentage_increase_l2373_237326

theorem percentage_increase (B C : ℝ) (h1 : C = B - 30) : 
  let A := 3 * B
  100 * (A - C) / C = 200 + 9000 / C := by
  sorry

end percentage_increase_l2373_237326


namespace dot_product_result_l2373_237398

theorem dot_product_result :
  let a : ℝ × ℝ := (2 * Real.sin (35 * π / 180), 2 * Real.cos (35 * π / 180))
  let b : ℝ × ℝ := (Real.cos (5 * π / 180), -Real.sin (5 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_result_l2373_237398


namespace parabola_directrix_l2373_237302

/-- The equation of the directrix of the parabola y = x^2 is y = -1/4 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), y = k ∧ k = -1/4) :=
by sorry

end parabola_directrix_l2373_237302


namespace sphere_surface_area_of_circumscribed_rectangular_solid_l2373_237341

/-- The surface area of a sphere circumscribing a rectangular solid with dimensions √3, √2, and 1 is 6π. -/
theorem sphere_surface_area_of_circumscribed_rectangular_solid :
  let length : ℝ := Real.sqrt 3
  let width : ℝ := Real.sqrt 2
  let height : ℝ := 1
  let diagonal : ℝ := Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2)
  let radius : ℝ := diagonal / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 6 * Real.pi := by
sorry

end sphere_surface_area_of_circumscribed_rectangular_solid_l2373_237341


namespace problem_solution_l2373_237355

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem problem_solution :
  (∀ x : ℝ, |g x| < 5 ↔ x ∈ Set.Ioo (-1) 3) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
    a ∈ Set.Iic (-6) ∪ Set.Ici 0) :=
by sorry

end problem_solution_l2373_237355


namespace work_scaling_l2373_237315

theorem work_scaling (people₁ work₁ days : ℕ) (people₂ : ℕ) :
  people₁ > 0 →
  work₁ > 0 →
  days > 0 →
  (people₁ * work₁ = people₁ * people₁) →
  people₂ = people₁ * (people₂ / people₁) →
  (people₂ / people₁ : ℚ) * work₁ = people₂ / people₁ * people₁ :=
by sorry

end work_scaling_l2373_237315


namespace deleted_pictures_l2373_237331

theorem deleted_pictures (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 24)
  (h2 : museum_pics = 12)
  (h3 : remaining_pics = 22) :
  zoo_pics + museum_pics - remaining_pics = 14 := by
  sorry

end deleted_pictures_l2373_237331


namespace karls_savings_l2373_237358

/-- The problem of calculating Karl's savings --/
theorem karls_savings :
  let folder_price : ℚ := 5/2
  let pen_price : ℚ := 1
  let folder_count : ℕ := 7
  let pen_count : ℕ := 10
  let folder_discount : ℚ := 3/10
  let pen_discount : ℚ := 15/100
  
  let folder_savings := folder_count * (folder_price * folder_discount)
  let pen_savings := pen_count * (pen_price * pen_discount)
  
  folder_savings + pen_savings = 27/4 := by
  sorry

end karls_savings_l2373_237358


namespace quadratic_inequality_l2373_237314

theorem quadratic_inequality (a b : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0) :
  ∃ n : ℤ, |n^2 + a*n + b| ≤ max (1/4 : ℝ) ((1/2 : ℝ) * Real.sqrt (a^2 - 4*b)) := by
  sorry

end quadratic_inequality_l2373_237314


namespace perpendicular_vectors_x_value_l2373_237343

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (x, 3)) 
  (hb : b = (2, x - 5)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 := by
sorry

end perpendicular_vectors_x_value_l2373_237343


namespace school_class_average_difference_l2373_237318

theorem school_class_average_difference :
  let total_students : ℕ := 200
  let total_teachers : ℕ := 5
  let class_sizes : List ℕ := [80, 60, 40, 15, 5]
  
  let t : ℚ := (class_sizes.sum : ℚ) / total_teachers
  
  let s : ℚ := (class_sizes.map (λ size => size * size)).sum / total_students
  
  t - s = -19.25 := by sorry

end school_class_average_difference_l2373_237318


namespace ratio_difference_l2373_237305

theorem ratio_difference (a b c : ℝ) (h1 : a / b = 3 / 5) (h2 : b / c = 5 / 7) (h3 : c = 56) : c - a = 32 := by
  sorry

end ratio_difference_l2373_237305


namespace no_14_cents_combination_l2373_237374

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A selection of coins is represented as a list of Coins -/
def CoinSelection := List Coin

/-- Calculates the total value of a coin selection in cents -/
def totalValue (selection : CoinSelection) : ℕ :=
  selection.map coinValue |>.sum

/-- Theorem stating that it's impossible to select 6 coins totaling 14 cents -/
theorem no_14_cents_combination :
  ∀ (selection : CoinSelection),
    selection.length = 6 →
    totalValue selection ≠ 14 :=
by sorry

end no_14_cents_combination_l2373_237374


namespace division_result_l2373_237336

theorem division_result : (3486 : ℝ) / 189 = 18.444444444444443 := by
  sorry

end division_result_l2373_237336


namespace bobby_shoe_count_bobby_shoe_count_proof_l2373_237301

/-- Given the relationships between Bonny's, Becky's, and Bobby's shoe counts, 
    prove that Bobby has 27 pairs of shoes. -/
theorem bobby_shoe_count : ℕ → ℕ → Prop :=
  fun becky_shoes bobby_shoes =>
    -- Bonny has 13 pairs of shoes
    -- Bonny's shoe count is 5 less than twice Becky's
    13 = 2 * becky_shoes - 5 →
    -- Bobby has 3 times as many shoes as Becky
    bobby_shoes = 3 * becky_shoes →
    -- Prove that Bobby has 27 pairs of shoes
    bobby_shoes = 27

/-- Proof of the theorem -/
theorem bobby_shoe_count_proof : ∃ (becky_shoes : ℕ), bobby_shoe_count becky_shoes 27 := by
  sorry

end bobby_shoe_count_bobby_shoe_count_proof_l2373_237301


namespace solve_for_k_l2373_237339

theorem solve_for_k (x y k : ℝ) : 
  x = -3 ∧ y = 2 ∧ 2 * x + k * y = 6 → k = 6 := by
  sorry

end solve_for_k_l2373_237339


namespace tickets_spent_on_beanie_l2373_237392

theorem tickets_spent_on_beanie (initial_tickets : Real) (lost_tickets : Real) (remaining_tickets : Real)
  (h1 : initial_tickets = 49.0)
  (h2 : lost_tickets = 6.0)
  (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by sorry

end tickets_spent_on_beanie_l2373_237392


namespace first_sequence_general_term_second_sequence_general_term_l2373_237323

/-- First sequence -/
def S₁ (n : ℕ) : ℚ := n^2 + (1/2) * n

/-- Second sequence -/
def S₂ (n : ℕ) : ℚ := (1/4) * n^2 + (2/3) * n + 3

/-- General term of the first sequence -/
def a₁ (n : ℕ) : ℚ := 2 * n - 1/2

/-- General term of the second sequence -/
def a₂ (n : ℕ) : ℚ :=
  if n = 1 then 47/12 else (6 * n + 5) / 12

theorem first_sequence_general_term (n : ℕ) :
  S₁ (n + 1) - S₁ n = a₁ (n + 1) :=
sorry

theorem second_sequence_general_term (n : ℕ) :
  S₂ (n + 1) - S₂ n = a₂ (n + 1) :=
sorry

end first_sequence_general_term_second_sequence_general_term_l2373_237323


namespace cone_volume_l2373_237394

/-- A cone with surface area π and lateral surface that unfolds into a semicircle has volume π/9 -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = 2 * r →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = π →  -- surface area is π
  h^2 + r^2 = l^2 →  -- Pythagorean theorem for cone
  (1/3) * π * r^2 * h = π/9 := by
sorry


end cone_volume_l2373_237394


namespace unique_students_count_unique_students_is_34_l2373_237330

/-- The number of unique students in a mathematics contest at Gauss High School --/
theorem unique_students_count : ℕ :=
  let euclid_class : ℕ := 12
  let raman_class : ℕ := 10
  let pythagoras_class : ℕ := 15
  let euclid_raman_overlap : ℕ := 3
  euclid_class + raman_class + pythagoras_class - euclid_raman_overlap

/-- Proof that the number of unique students is 34 --/
theorem unique_students_is_34 : unique_students_count = 34 := by
  sorry

end unique_students_count_unique_students_is_34_l2373_237330


namespace geraldo_tea_consumption_l2373_237384

-- Define the conversion factor from gallons to pints
def gallons_to_pints : ℝ := 8

-- Define the total amount of tea in gallons
def total_tea : ℝ := 20

-- Define the number of containers
def num_containers : ℝ := 80

-- Define the number of containers Geraldo drank
def containers_drunk : ℝ := 3.5

-- Theorem statement
theorem geraldo_tea_consumption :
  (total_tea / num_containers) * containers_drunk * gallons_to_pints = 7 := by
  sorry

end geraldo_tea_consumption_l2373_237384


namespace complex_modulus_equality_l2373_237375

theorem complex_modulus_equality : 
  Complex.abs ((7 - 5*Complex.I)*(3 + 4*Complex.I) + (4 - 3*Complex.I)*(2 + 7*Complex.I)) = Real.sqrt 6073 := by
  sorry

end complex_modulus_equality_l2373_237375


namespace airport_distance_is_130_l2373_237378

/-- Represents the problem of calculating the distance to the airport --/
def AirportDistance (initial_speed : ℝ) (speed_increase : ℝ) (initial_delay : ℝ) (actual_early : ℝ) : Prop :=
  ∃ (distance : ℝ) (time : ℝ),
    distance = initial_speed * (time + 1) ∧
    distance - initial_speed = (initial_speed + speed_increase) * (time - actual_early) ∧
    distance = 130

/-- The theorem stating that the distance to the airport is 130 miles --/
theorem airport_distance_is_130 :
  AirportDistance 40 20 1 0.25 := by
  sorry

end airport_distance_is_130_l2373_237378


namespace triangle_area_change_l2373_237327

/-- Theorem: Effect on triangle area when height is decreased by 40% and base is increased by 40% -/
theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = base * 1.4)
  (h2 : height_new = height * 0.6)
  (h3 : area = (base * height) / 2)
  (h4 : area_new = (base_new * height_new) / 2) :
  area_new = area * 0.84 := by
sorry

end triangle_area_change_l2373_237327


namespace stingray_count_shark_stingray_relation_total_fish_count_l2373_237309

/-- The number of stingrays in an aquarium -/
def num_stingrays : ℕ := 28

/-- The number of sharks in the aquarium -/
def num_sharks : ℕ := 2 * num_stingrays

/-- The total number of fish in the aquarium -/
def total_fish : ℕ := 84

/-- Theorem stating that the number of stingrays is 28 -/
theorem stingray_count : num_stingrays = 28 := by
  sorry

/-- Theorem verifying the relationship between sharks and stingrays -/
theorem shark_stingray_relation : num_sharks = 2 * num_stingrays := by
  sorry

/-- Theorem verifying the total number of fish -/
theorem total_fish_count : num_stingrays + num_sharks = total_fish := by
  sorry

end stingray_count_shark_stingray_relation_total_fish_count_l2373_237309


namespace operation_problem_l2373_237317

-- Define the set of operations
inductive Operation
| Add
| Sub
| Mul
| Div

-- Define the function that applies the operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (star mul : Operation) (h : apply_op star 16 4 / apply_op mul 8 2 = 4) :
  apply_op star 9 3 / apply_op mul 18 6 = 9 / 4 := by
  sorry

end operation_problem_l2373_237317


namespace nth_odd_multiple_of_three_l2373_237390

theorem nth_odd_multiple_of_three (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, k > 0 ∧ k = 6 * n - 3 ∧ k % 2 = 1 ∧ k % 3 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m < k ∧ m % 2 = 1 ∧ m % 3 = 0 → 
   ∃ i : ℕ, i < n ∧ m = 6 * i - 3) :=
by sorry

end nth_odd_multiple_of_three_l2373_237390


namespace A_star_B_equality_l2373_237332

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x ≥ 1}

def star_operation (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

theorem A_star_B_equality : 
  star_operation A B = {x : ℝ | (0 ≤ x ∧ x < 1) ∨ x > 3} :=
by sorry

end A_star_B_equality_l2373_237332


namespace range_of_a_l2373_237362

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 
  (0 < a ∧ a ≤ 1/3) :=
sorry

end range_of_a_l2373_237362


namespace max_elements_l2373_237389

structure RelationSystem where
  S : Type
  rel : S → S → Prop
  distinct_relation : ∀ a b : S, a ≠ b → (rel a b ∨ rel b a) ∧ ¬(rel a b ∧ rel b a)
  transitivity : ∀ a b c : S, a ≠ b → b ≠ c → a ≠ c → rel a b → rel b c → rel c a

theorem max_elements (R : RelationSystem) : 
  ∃ (n : ℕ), ∀ (m : ℕ), (∃ (f : Fin m → R.S), Function.Injective f) → m ≤ n :=
sorry

end max_elements_l2373_237389


namespace triangle_angle_calculation_l2373_237380

theorem triangle_angle_calculation (A B C a b c : Real) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  a * Real.cos B - b * Real.cos A = c →
  -- Given angle C
  C = π / 5 →
  -- Conclusion
  B = 3 * π / 10 := by
sorry

end triangle_angle_calculation_l2373_237380


namespace complex_product_real_implies_a_equals_one_l2373_237383

theorem complex_product_real_implies_a_equals_one (a : ℝ) :
  ((1 + Complex.I) * (1 - a * Complex.I)).im = 0 → a = 1 := by
  sorry

end complex_product_real_implies_a_equals_one_l2373_237383


namespace sum_six_consecutive_integers_l2373_237349

theorem sum_six_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end sum_six_consecutive_integers_l2373_237349


namespace complex_imaginary_x_value_l2373_237393

/-- A complex number z is imaginary if its real part is zero -/
def IsImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_imaginary_x_value (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 1) (x + 1)
  IsImaginary z → x = 1 := by
  sorry

end complex_imaginary_x_value_l2373_237393


namespace cake_problem_l2373_237353

/-- Proves that the initial number of cakes is 12, given the conditions of the problem. -/
theorem cake_problem (total : ℕ) (fallen : ℕ) (undamaged : ℕ) (destroyed : ℕ) 
  (h1 : fallen = total / 2)
  (h2 : undamaged = fallen / 2)
  (h3 : destroyed = 3)
  (h4 : fallen = undamaged + destroyed) :
  total = 12 := by
  sorry

end cake_problem_l2373_237353


namespace quadratic_inequality_problem_l2373_237373

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5*x - 2

-- Define the solution set of the original inequality
def solution_set (a : ℝ) := {x : ℝ | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) := a * x^2 - 5*x + a^2 - 1

-- Theorem statement
theorem quadratic_inequality_problem 
  (a : ℝ) 
  (h : ∀ x, f a x > 0 ↔ x ∈ solution_set a) :
  a = -2 ∧ 
  (∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end quadratic_inequality_problem_l2373_237373


namespace geometric_sequence_103rd_term_l2373_237356

/-- Given a geometric sequence with first term a and common ratio r,
    this function returns the nth term of the sequence. -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem geometric_sequence_103rd_term :
  let a : ℝ := 4
  let r : ℝ := -3
  geometric_sequence a r 103 = 4 * 3^102 := by
sorry

end geometric_sequence_103rd_term_l2373_237356


namespace nut_storage_impact_l2373_237311

/-- Represents the types of nuts found in Mason's car -/
inductive NutType
  | Almond
  | Walnut
  | Hazelnut

/-- Represents the squirrels and their nut-storing behavior -/
structure Squirrel where
  nutType : NutType
  count : Nat
  nutsPerDay : Nat
  days : Nat

/-- Calculates the total number of nuts stored by a group of squirrels -/
def totalNuts (s : Squirrel) : Nat :=
  s.count * s.nutsPerDay * s.days

/-- Calculates the weight of a single nut in grams -/
def nutWeight (n : NutType) : Rat :=
  match n with
  | NutType.Almond => 1/2
  | NutType.Walnut => 10
  | NutType.Hazelnut => 2

/-- Calculates the total weight of nuts stored by a group of squirrels -/
def totalWeight (s : Squirrel) : Rat :=
  (totalNuts s : Rat) * nutWeight s.nutType

/-- Calculates the efficiency reduction based on the total weight of nuts -/
def efficiencyReduction (totalWeight : Rat) : Rat :=
  min 100 (totalWeight / 100)

/-- The main theorem stating the total weight of nuts and efficiency reduction -/
theorem nut_storage_impact (almondSquirrels walnutSquirrels hazelnutSquirrels : Squirrel) 
    (h1 : almondSquirrels = ⟨NutType.Almond, 2, 30, 35⟩)
    (h2 : walnutSquirrels = ⟨NutType.Walnut, 3, 20, 40⟩)
    (h3 : hazelnutSquirrels = ⟨NutType.Hazelnut, 1, 10, 45⟩) :
    totalWeight almondSquirrels + totalWeight walnutSquirrels + totalWeight hazelnutSquirrels = 25950 ∧
    efficiencyReduction (totalWeight almondSquirrels + totalWeight walnutSquirrels + totalWeight hazelnutSquirrels) = 100 := by
  sorry


end nut_storage_impact_l2373_237311


namespace exists_x_where_exp_leq_x_plus_one_l2373_237333

theorem exists_x_where_exp_leq_x_plus_one : ∃ x : ℝ, Real.exp x ≤ x + 1 := by
  sorry

end exists_x_where_exp_leq_x_plus_one_l2373_237333


namespace trapezoid_side_length_l2373_237399

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC is 4:1
  area_ratio : ab / cd = 4
  -- The sum of AB and CD is 250
  sum_sides : ab + cd = 250

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC 
to the area of triangle ADC is 4:1, and AB + CD = 250 cm, then AB = 200 cm.
-/
theorem trapezoid_side_length (t : Trapezoid) : t.ab = 200 := by
  sorry

end trapezoid_side_length_l2373_237399


namespace megan_initial_cupcakes_l2373_237337

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 43

/-- The number of packages Megan could make with the remaining cupcakes -/
def num_packages : ℕ := 4

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 7

/-- The initial number of cupcakes Megan baked -/
def initial_cupcakes : ℕ := todd_ate + num_packages * cupcakes_per_package

theorem megan_initial_cupcakes : initial_cupcakes = 71 := by
  sorry

end megan_initial_cupcakes_l2373_237337


namespace painted_cube_theorem_l2373_237379

/-- Represents a painted cube that can be cut into smaller cubes -/
structure PaintedCube where
  edge : ℕ  -- Edge length of the large cube
  small_edge : ℕ  -- Edge length of the smaller cubes

/-- Counts the number of smaller cubes with exactly one painted face -/
def count_one_face_painted (cube : PaintedCube) : ℕ :=
  6 * (cube.edge - 2) * (cube.edge - 2)

/-- Counts the number of smaller cubes with exactly two painted faces -/
def count_two_faces_painted (cube : PaintedCube) : ℕ :=
  12 * (cube.edge - 2)

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.edge = 10) 
  (h2 : cube.small_edge = 1) : 
  count_one_face_painted cube = 384 ∧ count_two_faces_painted cube = 96 := by
  sorry

#eval count_one_face_painted ⟨10, 1⟩
#eval count_two_faces_painted ⟨10, 1⟩

end painted_cube_theorem_l2373_237379


namespace problem_solution_l2373_237386

def U : Set ℕ := {2, 3, 4, 5, 6}

def A : Set ℕ := {x ∈ U | x^2 - 6*x + 8 = 0}

def B : Set ℕ := {2, 5, 6}

theorem problem_solution : (U \ A) ∪ B = {2, 3, 5, 6} := by
  sorry

end problem_solution_l2373_237386


namespace median_is_90_l2373_237334

/-- Represents the score distribution of students -/
structure ScoreDistribution where
  score_70 : Nat
  score_80 : Nat
  score_90 : Nat
  score_100 : Nat

/-- Calculates the total number of students -/
def total_students (sd : ScoreDistribution) : Nat :=
  sd.score_70 + sd.score_80 + sd.score_90 + sd.score_100

/-- Defines the median score for a given score distribution -/
def median_score (sd : ScoreDistribution) : Nat :=
  if sd.score_70 + sd.score_80 ≥ (total_students sd + 1) / 2 then 80
  else if sd.score_70 + sd.score_80 + sd.score_90 ≥ (total_students sd + 1) / 2 then 90
  else 100

/-- Theorem stating that the median score for the given distribution is 90 -/
theorem median_is_90 (sd : ScoreDistribution) 
  (h1 : sd.score_70 = 1)
  (h2 : sd.score_80 = 6)
  (h3 : sd.score_90 = 5)
  (h4 : sd.score_100 = 3) :
  median_score sd = 90 := by
  sorry

end median_is_90_l2373_237334


namespace F_of_2_f_of_3_equals_341_l2373_237312

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- Theorem statement
theorem F_of_2_f_of_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end F_of_2_f_of_3_equals_341_l2373_237312


namespace m_greater_than_n_l2373_237322

theorem m_greater_than_n (a : ℝ) : 2 * a * (a - 2) + 7 > (a - 2) * (a - 3) := by
  sorry

end m_greater_than_n_l2373_237322
