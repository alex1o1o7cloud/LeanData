import Mathlib

namespace incorrect_observation_value_l3526_352659

theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h_n : n = 50) 
  (h_original_mean : original_mean = 36) 
  (h_new_mean : new_mean = 36.02) 
  (h_correct_value : correct_value = 48) :
  ∃ (incorrect_value : ℝ), 
    n * new_mean = n * original_mean - incorrect_value + correct_value ∧ 
    incorrect_value = 47 := by
  sorry

end incorrect_observation_value_l3526_352659


namespace dogwood_trees_in_other_part_l3526_352677

/-- The number of dogwood trees in the first part of the park -/
def trees_in_first_part : ℝ := 5.0

/-- The number of trees park workers plan to cut down -/
def planned_trees_to_cut : ℝ := 7.0

/-- The number of park workers on the job -/
def park_workers : ℝ := 8.0

/-- The number of dogwood trees left in the park after the work is done -/
def trees_left_after_work : ℝ := 2.0

/-- The number of dogwood trees in the other part of the park -/
def trees_in_other_part : ℝ := trees_left_after_work

theorem dogwood_trees_in_other_part : 
  trees_in_other_part = 2.0 :=
by sorry

end dogwood_trees_in_other_part_l3526_352677


namespace inequality_solution_l3526_352680

theorem inequality_solution : 
  ∃! (n : ℕ), n ≥ 3 ∧ 
  (∀ (x : ℝ), x ≥ 3 → 
    (Real.sqrt (5 * x - 11) - Real.sqrt (5 * x^2 - 21 * x + 21) ≥ 5 * x^2 - 26 * x + 32) →
    x = n) := by
  sorry

end inequality_solution_l3526_352680


namespace num_lineups_eq_2277_l3526_352606

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def special_players : ℕ := 3

/-- The number of possible lineups given the constraints -/
def num_lineups : ℕ :=
  3 * (Nat.choose (team_size - special_players) (lineup_size - 1)) +
  Nat.choose (team_size - special_players) lineup_size

/-- Theorem stating that the number of possible lineups is 2277 -/
theorem num_lineups_eq_2277 : num_lineups = 2277 := by
  sorry

end num_lineups_eq_2277_l3526_352606


namespace least_number_with_remainder_l3526_352660

theorem least_number_with_remainder (n : ℕ) : n = 266 ↔ 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 33 ≠ 2 ∨ m % 8 ≠ 2)) ∧ 
  n % 33 = 2 ∧ n % 8 = 2 :=
sorry

end least_number_with_remainder_l3526_352660


namespace altitudes_5_12_13_impossible_l3526_352625

-- Define a function to check if three numbers can be altitudes of a triangle
def canBeAltitudes (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    x * a = y * b ∧ y * b = z * c →
    x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem altitudes_5_12_13_impossible :
  ¬ canBeAltitudes 5 12 13 :=
sorry

end altitudes_5_12_13_impossible_l3526_352625


namespace fractional_equation_solution_range_l3526_352605

/-- Given a fractional equation (2x - m) / (x + 1) = 3 where x is positive,
    prove that m < -3 -/
theorem fractional_equation_solution_range (x m : ℝ) :
  (2 * x - m) / (x + 1) = 3 → x > 0 → m < -3 := by
  sorry

end fractional_equation_solution_range_l3526_352605


namespace cat_weight_ratio_l3526_352670

def megs_cat_weight : ℕ := 20
def weight_difference : ℕ := 8

def annes_cat_weight : ℕ := megs_cat_weight + weight_difference

theorem cat_weight_ratio :
  (megs_cat_weight : ℚ) / annes_cat_weight = 5 / 7 := by sorry

end cat_weight_ratio_l3526_352670


namespace union_of_sets_l3526_352674

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  M ∪ N = {1, 2, 3} := by sorry

end union_of_sets_l3526_352674


namespace min_value_theorem_l3526_352632

theorem min_value_theorem (m n p x y z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0)
  (hmnp : m * n * p = 8)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x * y * z = 8) :
  ∃ (min : ℝ), min = 12 + 4 * (m + n + p) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 8 →
    a^2 + b^2 + c^2 + m*a*b + n*a*c + p*b*c ≥ min :=
by sorry

end min_value_theorem_l3526_352632


namespace investment_return_calculation_l3526_352623

theorem investment_return_calculation (total_investment small_investment large_investment : ℝ)
  (combined_return_rate small_return_rate : ℝ) :
  total_investment = small_investment + large_investment →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.085 →
  small_return_rate = 0.07 →
  (combined_return_rate * total_investment = small_return_rate * small_investment + 
    (large_investment * 0.09)) :=
by sorry

end investment_return_calculation_l3526_352623


namespace quadratic_root_distance_translation_l3526_352630

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0 and two distinct roots
    a distance p apart, the downward translation needed to make the distance
    between the roots 2p is (3b^2)/(4a) - 3c. -/
theorem quadratic_root_distance_translation
  (a b c p : ℝ)
  (h_a_pos : a > 0)
  (h_distinct_roots : ∃ x y, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_distance : ∃ x y, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ |x - y| = p) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * x^2 + b * x + (c - ((3 * b^2) / (4 * a) - 3 * c))
  ∃ x y, x ≠ y ∧ g x = 0 ∧ g y = 0 ∧ |x - y| = 2 * p :=
sorry

end quadratic_root_distance_translation_l3526_352630


namespace triangle_circumradius_l3526_352636

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √3 and c - 2b + 2√3 cos C = 0, then the radius of the circumcircle is 1. -/
theorem triangle_circumradius (a b c A B C : ℝ) : 
  a = Real.sqrt 3 →
  c - 2*b + 2*(Real.sqrt 3)*(Real.cos C) = 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (a / (2 * Real.sin A)) = 1 := by
  sorry

end triangle_circumradius_l3526_352636


namespace daisy_toys_count_l3526_352658

def monday_toys : ℕ := 5
def tuesday_toys_left : ℕ := 3
def tuesday_toys_bought : ℕ := 3
def wednesday_toys_bought : ℕ := 5

def total_toys : ℕ := monday_toys + (monday_toys - tuesday_toys_left) + tuesday_toys_bought + wednesday_toys_bought

theorem daisy_toys_count : total_toys = 15 := by sorry

end daisy_toys_count_l3526_352658


namespace smallest_k_for_800_digit_sum_l3526_352685

/-- Represents a number consisting of k digits of 7 -/
def seventySevenNumber (k : ℕ) : ℕ :=
  (7 * (10^k - 1)) / 9

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem smallest_k_for_800_digit_sum :
  ∃ k : ℕ, k = 88 ∧
  (∀ m : ℕ, m < k → sumOfDigits (5 * seventySevenNumber m) < 800) ∧
  sumOfDigits (5 * seventySevenNumber k) = 800 :=
sorry

end smallest_k_for_800_digit_sum_l3526_352685


namespace vincent_laundry_theorem_l3526_352622

/-- Represents the types of laundry loads --/
inductive LoadType
  | Regular
  | Delicate
  | Heavy

/-- Represents a day's laundry schedule --/
structure DaySchedule where
  regular : Nat
  delicate : Nat
  heavy : Nat

/-- Calculate total loads for a day --/
def totalLoads (schedule : DaySchedule) : Nat :=
  schedule.regular + schedule.delicate + schedule.heavy

/-- Vincent's laundry week --/
def laundryWeek : List DaySchedule :=
  [
    { regular := 2, delicate := 1, heavy := 3 },  -- Wednesday
    { regular := 4, delicate := 2, heavy := 4 },  -- Thursday
    { regular := 2, delicate := 1, heavy := 0 },  -- Friday
    { regular := 0, delicate := 0, heavy := 1 }   -- Saturday
  ]

theorem vincent_laundry_theorem :
  (laundryWeek.map totalLoads).sum = 20 := by
  sorry

#eval (laundryWeek.map totalLoads).sum

end vincent_laundry_theorem_l3526_352622


namespace sum_of_ratios_is_four_l3526_352678

/-- Given two nonconstant geometric sequences with common first term,
    if the difference of their third terms is four times the difference of their second terms,
    then the sum of their common ratios is 4. -/
theorem sum_of_ratios_is_four (k p r : ℝ) (h_k : k ≠ 0) (h_p : p ≠ 1) (h_r : r ≠ 1) 
    (h_eq : k * p^2 - k * r^2 = 4 * (k * p - k * r)) :
  p + r = 4 := by
sorry

end sum_of_ratios_is_four_l3526_352678


namespace gcd_lcm_sum_36_2310_l3526_352691

theorem gcd_lcm_sum_36_2310 : Nat.gcd 36 2310 + Nat.lcm 36 2310 = 13866 := by
  sorry

end gcd_lcm_sum_36_2310_l3526_352691


namespace polynomial_evaluation_l3526_352617

theorem polynomial_evaluation : 7^4 - 4 * 7^3 + 6 * 7^2 - 4 * 7 + 1 = 1296 := by
  sorry

end polynomial_evaluation_l3526_352617


namespace population_increase_theorem_l3526_352663

-- Define the increase factors
def increase_factor_0_to_2 : ℝ := 1.1
def increase_factor_2_to_5 : ℝ := 1.2

-- Define the total increase factor
def total_increase_factor : ℝ := increase_factor_0_to_2 * increase_factor_2_to_5

-- Theorem statement
theorem population_increase_theorem :
  (total_increase_factor - 1) * 100 = 32 := by
  sorry


end population_increase_theorem_l3526_352663


namespace ladybugs_on_tuesday_l3526_352635

theorem ladybugs_on_tuesday (monday_ladybugs : ℕ) (dots_per_ladybug : ℕ) (total_dots : ℕ) :
  monday_ladybugs = 8 →
  dots_per_ladybug = 6 →
  total_dots = 78 →
  ∃ tuesday_ladybugs : ℕ, 
    tuesday_ladybugs = 5 ∧
    total_dots = monday_ladybugs * dots_per_ladybug + tuesday_ladybugs * dots_per_ladybug :=
by sorry

end ladybugs_on_tuesday_l3526_352635


namespace quadratic_polynomial_satisfies_conditions_l3526_352651

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = 1.1 * x^2 - 2.1 * x + 5) ∧
    q (-1) = 4 ∧
    q 2 = 1 ∧
    q 4 = 10 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l3526_352651


namespace binary_to_octal_l3526_352645

-- Define the binary number
def binary_number : ℕ := 0b110101

-- Define the octal number
def octal_number : ℕ := 66

-- Theorem stating that the binary number is equal to the octal number
theorem binary_to_octal : binary_number = octal_number := by
  sorry

end binary_to_octal_l3526_352645


namespace absolute_value_of_z_l3526_352608

theorem absolute_value_of_z (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : Complex.abs z = 1 := by
  sorry

end absolute_value_of_z_l3526_352608


namespace brians_trip_distance_l3526_352609

/-- Calculates the distance traveled given car efficiency and gas used -/
def distance_traveled (efficiency : ℝ) (gas_used : ℝ) : ℝ :=
  efficiency * gas_used

/-- Proves that given a car efficiency of 20 miles per gallon and 
    a gas usage of 3 gallons, the distance traveled is 60 miles -/
theorem brians_trip_distance : 
  distance_traveled 20 3 = 60 := by
  sorry

end brians_trip_distance_l3526_352609


namespace mixed_gender_selections_l3526_352668

-- Define the number of male and female students
def num_male_students : Nat := 5
def num_female_students : Nat := 3

-- Define the total number of students
def total_students : Nat := num_male_students + num_female_students

-- Define the number of students to be selected
def students_to_select : Nat := 3

-- Function to calculate combinations
def combination (n : Nat) (r : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem mixed_gender_selections :
  combination total_students students_to_select -
  combination num_male_students students_to_select -
  combination num_female_students students_to_select = 45 := by
  sorry


end mixed_gender_selections_l3526_352668


namespace condition_for_reciprocal_inequality_l3526_352683

theorem condition_for_reciprocal_inequality (a : ℝ) :
  (∀ a, (1 / a > 1 → a < 1)) ∧ (∃ a, a < 1 ∧ 1 / a ≤ 1) := by
  sorry

end condition_for_reciprocal_inequality_l3526_352683


namespace prism_volume_l3526_352612

theorem prism_volume (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a * b = 100 → a * h = 50 → b * h = 40 → h = 10 →
  a * b * h = 200 := by sorry

end prism_volume_l3526_352612


namespace blanket_donation_ratio_l3526_352649

/-- The ratio of blankets collected on the second day to the first day -/
def blanket_ratio (team_size : ℕ) (first_day_per_person : ℕ) (last_day_total : ℕ) (total_blankets : ℕ) : ℚ :=
  let first_day := team_size * first_day_per_person
  let second_day := total_blankets - first_day - last_day_total
  (second_day : ℚ) / first_day

/-- Proves that the ratio of blankets collected on the second day to the first day is 3 -/
theorem blanket_donation_ratio :
  blanket_ratio 15 2 22 142 = 3 := by
  sorry

end blanket_donation_ratio_l3526_352649


namespace surface_area_of_sliced_solid_l3526_352615

/-- Right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Point on an edge of the prism -/
structure EdgePoint where
  edge : Fin 3  -- 0: AC, 1: BC, 2: DC
  position : ℝ  -- Fraction of the way along the edge

/-- Solid formed by slicing the prism -/
def SlicedSolid (prism : RightPrism) (p q r : EdgePoint) : Type := sorry

/-- Surface area of a sliced solid -/
noncomputable def surfaceArea (prism : RightPrism) (solid : SlicedSolid prism p q r) : ℝ := sorry

/-- The main theorem -/
theorem surface_area_of_sliced_solid 
  (prism : RightPrism)
  (h_height : prism.height = 20)
  (h_base : prism.baseSideLength = 10)
  (p : EdgePoint)
  (h_p : p.edge = 0 ∧ p.position = 1/3)
  (q : EdgePoint)
  (h_q : q.edge = 1 ∧ q.position = 1/3)
  (r : EdgePoint)
  (h_r : r.edge = 2 ∧ r.position = 1/2)
  (solid : SlicedSolid prism p q r) :
  surfaceArea prism solid = (50 * Real.sqrt 3 + 25 * Real.sqrt 2 / 3 + 50 * Real.sqrt 10) / 3 := by
  sorry

end surface_area_of_sliced_solid_l3526_352615


namespace a7_not_prime_l3526_352626

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Defines the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 170  -- Initial value
  | n + 1 => a n + reverseDigits (a n)

/-- States that a_7 is not prime -/
theorem a7_not_prime : ¬ Nat.Prime (a 7) := by sorry

end a7_not_prime_l3526_352626


namespace inequality_proof_l3526_352602

theorem inequality_proof (x : ℝ) (n : ℕ) 
  (h1 : |x| < 1) (h2 : n ≥ 2) : (1 + x)^n + (1 - x)^n < 2^n := by
  sorry

end inequality_proof_l3526_352602


namespace extreme_values_range_c_l3526_352629

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 2*c*x^2 + x

-- Define the property of having extreme values
def has_extreme_values (c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (3*x₁^2 - 4*c*x₁ + 1 = 0) ∧ 
    (3*x₂^2 - 4*c*x₂ + 1 = 0)

-- State the theorem
theorem extreme_values_range_c :
  ∀ c : ℝ, has_extreme_values c ↔ (c < -Real.sqrt 3 / 2 ∨ c > Real.sqrt 3 / 2) :=
by sorry

end extreme_values_range_c_l3526_352629


namespace crabby_squido_ratio_l3526_352654

def squido_oysters : ℕ := 200
def total_oysters : ℕ := 600

def crabby_oysters : ℕ := total_oysters - squido_oysters

theorem crabby_squido_ratio : 
  (crabby_oysters : ℚ) / squido_oysters = 2 := by sorry

end crabby_squido_ratio_l3526_352654


namespace power_function_m_value_l3526_352627

theorem power_function_m_value : ∃! m : ℝ, m^2 - 9*m + 19 = 1 ∧ 2*m^2 - 7*m - 9 ≤ 0 := by
  sorry

end power_function_m_value_l3526_352627


namespace bungee_cord_extension_l3526_352695

/-- The maximum extension of a bungee cord in a bungee jumping scenario -/
theorem bungee_cord_extension
  (m : ℝ) -- mass of the person
  (H : ℝ) -- maximum fall distance
  (k : ℝ) -- spring constant of the bungee cord
  (L₀ : ℝ) -- original length of the bungee cord
  (h : ℝ) -- extension of the bungee cord
  (g : ℝ) -- gravitational acceleration
  (hpos : h > 0)
  (mpos : m > 0)
  (kpos : k > 0)
  (Hpos : H > 0)
  (L₀pos : L₀ > 0)
  (gpos : g > 0)
  (hooke : k * h = 4 * m * g) -- Hooke's law and maximum tension condition
  (energy : m * g * H = (1/2) * k * h^2) -- Conservation of energy
  : h = H / 2 := by
  sorry

end bungee_cord_extension_l3526_352695


namespace min_PM_AB_implies_AB_equation_l3526_352611

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent points A and B on circle M
def tangent_points (xA yA xB yB : ℝ) : Prop :=
  circle_M xA yA ∧ circle_M xB yB

-- Define the minimization condition
def min_condition (xP yP xM yM xA yA xB yB : ℝ) : Prop :=
  ∀ x y, point_P x y →
    (x - xM)^2 + (y - yM)^2 ≤ (xP - xM)^2 + (yP - yM)^2

-- Theorem statement
theorem min_PM_AB_implies_AB_equation :
  ∀ xP yP xM yM xA yA xB yB,
    point_P xP yP →
    tangent_points xA yA xB yB →
    min_condition xP yP xM yM xA yA xB yB →
    2*xA + yA + 1 = 0 ∧ 2*xB + yB + 1 = 0 :=
sorry

end min_PM_AB_implies_AB_equation_l3526_352611


namespace parallel_lines_m_value_l3526_352604

theorem parallel_lines_m_value (x y : ℝ) :
  (∀ x y, 2*x + 3*y + 1 = 0 ↔ m*x + 6*y - 5 = 0) →
  m = 4 :=
by sorry

end parallel_lines_m_value_l3526_352604


namespace water_balloon_count_l3526_352637

/-- The total number of filled water balloons Max and Zach have -/
def total_filled_balloons (max_time max_rate zach_time zach_rate popped : ℕ) : ℕ :=
  max_time * max_rate + zach_time * zach_rate - popped

/-- Theorem: The total number of filled water balloons Max and Zach have is 170 -/
theorem water_balloon_count : total_filled_balloons 30 2 40 3 10 = 170 := by
  sorry

end water_balloon_count_l3526_352637


namespace animal_shelter_cats_l3526_352633

theorem animal_shelter_cats (dogs : ℕ) (initial_ratio_dogs initial_ratio_cats : ℕ) 
  (final_ratio_dogs final_ratio_cats : ℕ) (additional_cats : ℕ) : 
  dogs = 75 →
  initial_ratio_dogs = 15 →
  initial_ratio_cats = 7 →
  final_ratio_dogs = 15 →
  final_ratio_cats = 11 →
  (dogs : ℚ) / (dogs * initial_ratio_cats / initial_ratio_dogs : ℚ) = 
    initial_ratio_dogs / initial_ratio_cats →
  (dogs : ℚ) / (dogs * initial_ratio_cats / initial_ratio_dogs + additional_cats : ℚ) = 
    final_ratio_dogs / final_ratio_cats →
  additional_cats = 20 := by
  sorry

end animal_shelter_cats_l3526_352633


namespace inscribed_circle_radius_arithmetic_progression_l3526_352616

/-- Given a triangle with sides a, b, c forming an arithmetic progression,
    prove that the radius of the inscribed circle is one-third of the altitude to side b. -/
theorem inscribed_circle_radius_arithmetic_progression
  (a b c : ℝ) (h_order : a ≤ b ∧ b ≤ c) (h_arithmetic : 2 * b = a + c)
  (r : ℝ) (h_b : ℝ) (S : ℝ) 
  (h_area_inradius : 2 * S = r * (a + b + c))
  (h_area_altitude : 2 * S = h_b * b) :
  r = h_b / 3 := by sorry

end inscribed_circle_radius_arithmetic_progression_l3526_352616


namespace cone_lateral_surface_is_sector_l3526_352631

/-- Represents the possible shapes of an unfolded lateral surface of a cone -/
inductive UnfoldedShape
  | Triangle
  | Rectangle
  | Square
  | Sector

/-- Represents a cone -/
structure Cone where
  -- Add any necessary properties of a cone here

/-- The lateral surface of a cone when unfolded -/
def lateralSurface (c : Cone) : UnfoldedShape := sorry

/-- Theorem stating that the lateral surface of a cone, when unfolded, is shaped like a sector -/
theorem cone_lateral_surface_is_sector (c : Cone) : lateralSurface c = UnfoldedShape.Sector := by
  sorry

end cone_lateral_surface_is_sector_l3526_352631


namespace pirate_catch_time_l3526_352690

/-- Represents the pursuit problem between a pirate ship and a trading vessel -/
structure PursuitProblem where
  initial_distance : ℝ
  pirate_speed_initial : ℝ
  trader_speed : ℝ
  pursuit_start_time : ℝ
  speed_change_time : ℝ
  new_speed_ratio : ℝ

/-- Calculates the time at which the pirate ship catches the trading vessel -/
def catch_time (p : PursuitProblem) : ℝ :=
  sorry

/-- Theorem stating that the catch time for the given problem is 4:40 p.m. (16.67 hours) -/
theorem pirate_catch_time :
  let problem := PursuitProblem.mk 12 12 9 12 3 1.2
  catch_time problem = 16 + 2/3 :=
sorry

end pirate_catch_time_l3526_352690


namespace unit_conversions_l3526_352687

-- Define unit conversion factors
def cm_to_dm : ℚ := 1 / 10
def hectare_to_km2 : ℚ := 1 / 100
def yuan_to_jiao : ℚ := 10
def yuan_to_fen : ℚ := 100
def hectare_to_m2 : ℚ := 10000
def dm_to_m : ℚ := 1 / 10
def m_to_cm : ℚ := 100

-- Theorem statement
theorem unit_conversions :
  (70000 * cm_to_dm^2 = 700) ∧
  (800 * hectare_to_km2 = 8) ∧
  (1.65 * yuan_to_jiao = 16.5) ∧
  (400 * hectare_to_m2 = 4000000) ∧
  (0.57 * yuan_to_fen = 57) ∧
  (5000 * dm_to_m^2 = 50) ∧
  (60000 / hectare_to_m2 = 6) ∧
  (9 * m_to_cm = 900) :=
by sorry

end unit_conversions_l3526_352687


namespace cubic_roots_sum_cubes_l3526_352643

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (6 * a^3 + 500 * a + 1001 = 0) →
  (6 * b^3 + 500 * b + 1001 = 0) →
  (6 * c^3 + 500 * c + 1001 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := by
  sorry

end cubic_roots_sum_cubes_l3526_352643


namespace min_perimeter_triangle_l3526_352666

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sides_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircles of a triangle -/
def excircles (t : Triangle) : Fin 3 → Excircle := sorry

/-- Checks if two circles are internally tangent -/
def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem min_perimeter_triangle (t : Triangle) 
  (h1 : is_internally_tangent (incircle t) { center := (excircles t 0).center, radius := (excircles t 0).radius })
  (h2 : ¬ is_internally_tangent (incircle t) { center := (excircles t 1).center, radius := (excircles t 1).radius })
  (h3 : ¬ is_internally_tangent (incircle t) { center := (excircles t 2).center, radius := (excircles t 2).radius }) :
  t.a + t.b + t.c ≥ 12 := by
  sorry

end min_perimeter_triangle_l3526_352666


namespace math_test_difference_l3526_352689

theorem math_test_difference (total_questions word_problems addition_subtraction_problems steve_can_answer : ℕ) :
  total_questions = 45 →
  word_problems = 17 →
  addition_subtraction_problems = 28 →
  steve_can_answer = 38 →
  total_questions - steve_can_answer = 7 :=
by
  sorry

end math_test_difference_l3526_352689


namespace quadratic_form_ratio_l3526_352640

theorem quadratic_form_ratio (j : ℝ) : ∃ (c p q : ℝ),
  (6 * j^2 - 4 * j + 12 = c * (j + p)^2 + q) ∧ (q / p = -34) := by
  sorry

end quadratic_form_ratio_l3526_352640


namespace intersection_condition_l3526_352696

def A (a : ℝ) : Set ℝ := {3, Real.sqrt a}
def B (a : ℝ) : Set ℝ := {1, a}

theorem intersection_condition (a : ℝ) : A a ∩ B a = {a} → a = 0 ∨ a = 3 := by
  sorry

end intersection_condition_l3526_352696


namespace chernomor_salary_manipulation_l3526_352620

/-- Represents a salary proposal for a single month -/
structure SalaryProposal where
  warrior_salaries : Fin 33 → ℝ
  chernomor_salary : ℝ

/-- The voting function: returns true if the majority of warriors vote in favor -/
def majority_vote (current : SalaryProposal) (proposal : SalaryProposal) : Prop :=
  (Finset.filter (fun i => proposal.warrior_salaries i > current.warrior_salaries i) Finset.univ).card > 16

/-- The theorem stating that Chernomor can achieve his goal -/
theorem chernomor_salary_manipulation :
  ∃ (initial : SalaryProposal) (proposals : Fin 36 → SalaryProposal),
    (∀ i : Fin 35, majority_vote (proposals i) (proposals (i + 1))) ∧
    (proposals 35).chernomor_salary = 10 * initial.chernomor_salary ∧
    (∀ j : Fin 33, (proposals 35).warrior_salaries j ≤ initial.warrior_salaries j / 10) :=
sorry

end chernomor_salary_manipulation_l3526_352620


namespace eighth_term_value_l3526_352655

/-- The nth term of a geometric sequence -/
def geometric_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

/-- The 8th term of the specific geometric sequence -/
def eighth_term : ℚ :=
  geometric_term 3 (3/2) 8

theorem eighth_term_value : eighth_term = 6561 / 128 := by
  sorry

end eighth_term_value_l3526_352655


namespace arithmetic_geometric_mean_inequality_l3526_352675

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (h : a > b) :
  (a + b) / 2 > Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end arithmetic_geometric_mean_inequality_l3526_352675


namespace square_equality_l3526_352614

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by sorry

end square_equality_l3526_352614


namespace unique_outfits_count_l3526_352676

def number_of_shirts : ℕ := 10
def number_of_ties : ℕ := 8
def shirts_per_outfit : ℕ := 5
def ties_per_outfit : ℕ := 4

theorem unique_outfits_count : 
  (Nat.choose number_of_shirts shirts_per_outfit) * 
  (Nat.choose number_of_ties ties_per_outfit) = 17640 := by
  sorry

end unique_outfits_count_l3526_352676


namespace max_product_sum_300_l3526_352657

theorem max_product_sum_300 :
  (∃ (a b : ℤ), a + b = 300 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 300 → a * b ≤ 22500) :=
by sorry

end max_product_sum_300_l3526_352657


namespace parabola_vertex_on_axis_l3526_352642

/-- A parabola with equation y = x^2 - kx + k - 1 has its vertex on a coordinate axis if and only if k = 2 or k = 0 -/
theorem parabola_vertex_on_axis (k : ℝ) : 
  (∃ x y : ℝ, (y = x^2 - k*x + k - 1) ∧ 
    ((x = 0 ∧ y = k - 1) ∨ (y = 0 ∧ x = k/2)) ∧
    (∀ x' y' : ℝ, y' = x'^2 - k*x' + k - 1 → y' ≥ y)) ↔ 
  (k = 2 ∨ k = 0) :=
sorry

end parabola_vertex_on_axis_l3526_352642


namespace intersection_condition_l3526_352619

/-- Two curves intersect at exactly two distinct points -/
def HasTwoDistinctIntersections (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ (f x₁ y₁ ∧ g x₁ y₁) ∧ (f x₂ y₂ ∧ g x₂ y₂) ∧
  ∀ (x y : ℝ), (f x y ∧ g x y) → ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))

/-- The circle equation -/
def Circle (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = b^2

/-- The parabola equation -/
def Parabola (b : ℝ) (x y : ℝ) : Prop := y = -x^2 + b

theorem intersection_condition (b : ℝ) :
  HasTwoDistinctIntersections (Circle b) (Parabola b) ↔ b = 1/2 :=
sorry

end intersection_condition_l3526_352619


namespace triangle_area_is_2_sqrt_6_l3526_352644

/-- A triangle with integral sides and perimeter 12 --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 12
  triangle_ineq_ab : a + b > c
  triangle_ineq_bc : b + c > a
  triangle_ineq_ca : c + a > b

/-- The area of a triangle with integral sides and perimeter 12 is 2√6 --/
theorem triangle_area_is_2_sqrt_6 (t : Triangle) : 
  ∃ (area : ℝ), area = 2 * Real.sqrt 6 ∧ area = (t.a * t.b * Real.sin (π / 3)) / 2 := by
  sorry

end triangle_area_is_2_sqrt_6_l3526_352644


namespace article_cost_price_l3526_352664

/-- Given an article marked 15% above its cost price, sold at Rs. 462 with a discount of 25.603864734299517%, prove that the cost price of the article is Rs. 540. -/
theorem article_cost_price (cost_price : ℝ) : 
  let markup_percentage : ℝ := 0.15
  let selling_price : ℝ := 462
  let discount_percentage : ℝ := 25.603864734299517
  let marked_price : ℝ := cost_price * (1 + markup_percentage)
  let discounted_price : ℝ := marked_price * (1 - discount_percentage / 100)
  discounted_price = selling_price → cost_price = 540 := by
sorry

#eval (462 : ℚ) / (1 - 25.603864734299517 / 100) / 1.15

end article_cost_price_l3526_352664


namespace arithmetic_sequence_common_difference_l3526_352624

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (h1 : a 1 + a 5 = 10) -- First condition: a_1 + a_5 = 10
  (h2 : a 4 = 7) -- Second condition: a_4 = 7
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) -- Definition of arithmetic sequence
  : a 2 - a 1 = 2 := by
sorry

end arithmetic_sequence_common_difference_l3526_352624


namespace maryville_population_increase_l3526_352682

/-- The average number of people added each year in Maryville between 2000 and 2005 -/
def average_population_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating that the average population increase in Maryville between 2000 and 2005 is 3400 -/
theorem maryville_population_increase :
  average_population_increase 450000 467000 = 3400 := by
  sorry

end maryville_population_increase_l3526_352682


namespace prism_volume_l3526_352694

/-- A right rectangular prism with given face areas has a volume of 30 cubic inches -/
theorem prism_volume (l w h : ℝ) 
  (face1 : l * w = 10)
  (face2 : w * h = 15)
  (face3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end prism_volume_l3526_352694


namespace percentage_change_equivalence_l3526_352672

theorem percentage_change_equivalence (p q N : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 50) (hN : N > 0) :
  N * (1 + p / 100) * (1 - q / 100) < N ↔ p < (100 * q) / (100 - q) := by
  sorry

end percentage_change_equivalence_l3526_352672


namespace samantha_pet_food_difference_l3526_352618

/-- Proves that Samantha bought 49 more cans of cat food than dog and bird food combined. -/
theorem samantha_pet_food_difference : 
  let cat_packages : ℕ := 8
  let dog_packages : ℕ := 5
  let bird_packages : ℕ := 3
  let cat_cans_per_package : ℕ := 12
  let dog_cans_per_package : ℕ := 7
  let bird_cans_per_package : ℕ := 4
  let total_cat_cans := cat_packages * cat_cans_per_package
  let total_dog_cans := dog_packages * dog_cans_per_package
  let total_bird_cans := bird_packages * bird_cans_per_package
  total_cat_cans - (total_dog_cans + total_bird_cans) = 49 := by
  sorry

#eval 8 * 12 - (5 * 7 + 3 * 4)  -- Should output 49

end samantha_pet_food_difference_l3526_352618


namespace semicircle_radius_prove_semicircle_radius_l3526_352662

theorem semicircle_radius : ℝ → Prop :=
fun r : ℝ =>
  (3 * (2 * r) + 2 * 12 = 2 * (2 * r) + 22 + 16 + 22) → r = 18

theorem prove_semicircle_radius : semicircle_radius 18 := by
  sorry

end semicircle_radius_prove_semicircle_radius_l3526_352662


namespace intersection_of_specific_sets_l3526_352647

theorem intersection_of_specific_sets :
  let A : Set ℕ := {1, 2, 5}
  let B : Set ℕ := {1, 3, 5}
  A ∩ B = {1, 5} := by
sorry

end intersection_of_specific_sets_l3526_352647


namespace sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12_l3526_352692

theorem sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12 :
  ∀ x : ℝ, Real.sin (3 * x + π / 4) = Real.cos (3 * (x - π / 12)) := by
  sorry

end sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12_l3526_352692


namespace problem_statement_l3526_352652

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The problem statement -/
theorem problem_statement :
  floor ((2015^2 : ℝ) / (2013 * 2014) - (2013^2 : ℝ) / (2014 * 2015)) = 0 := by
  sorry

end problem_statement_l3526_352652


namespace complement_M_equals_expected_l3526_352673

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set Nat := {1, 2}

-- Define the complement of M with respect to U
def complement_M : Set Nat := U \ M

-- Theorem statement
theorem complement_M_equals_expected : complement_M = {3, 4, 5} := by
  sorry

end complement_M_equals_expected_l3526_352673


namespace min_value_of_function_equality_condition_l3526_352671

theorem min_value_of_function (x : ℝ) (h : x > 1) : 1 / (x - 1) + x ≥ 3 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 1) : 1 / (x - 1) + x = 3 ↔ x = 2 := by
  sorry

end min_value_of_function_equality_condition_l3526_352671


namespace boat_speed_in_still_water_l3526_352697

/-- Given a boat that travels 15 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 10 km/hr. -/
theorem boat_speed_in_still_water
  (along_stream : ℝ)
  (against_stream : ℝ)
  (h_along : along_stream = 15)
  (h_against : against_stream = 5) :
  (along_stream + against_stream) / 2 = 10 :=
by sorry

end boat_speed_in_still_water_l3526_352697


namespace opposite_of_three_l3526_352603

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

theorem opposite_of_three : opposite 3 = -3 := by
  sorry

end opposite_of_three_l3526_352603


namespace find_c_l3526_352667

theorem find_c : ∃ c : ℝ, 
  (∃ n : ℤ, Int.floor c = n ∧ 3 * (n : ℝ)^2 + 12 * (n : ℝ) - 27 = 0) ∧ 
  (let frac := c - Int.floor c
   4 * frac^2 - 12 * frac + 5 = 0) ∧
  (0 ≤ c - Int.floor c ∧ c - Int.floor c < 1) ∧
  c = -8.5 := by
  sorry

end find_c_l3526_352667


namespace complex_multiplication_l3526_352607

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l3526_352607


namespace wrapping_paper_cost_l3526_352610

-- Define the problem parameters
def shirtBoxesPerRoll : ℕ := 5
def xlBoxesPerRoll : ℕ := 3
def totalShirtBoxes : ℕ := 20
def totalXlBoxes : ℕ := 12
def totalCost : ℚ := 32

-- Define the theorem
theorem wrapping_paper_cost :
  let rollsForShirtBoxes := totalShirtBoxes / shirtBoxesPerRoll
  let rollsForXlBoxes := totalXlBoxes / xlBoxesPerRoll
  let totalRolls := rollsForShirtBoxes + rollsForXlBoxes
  totalCost / totalRolls = 4 := by sorry

end wrapping_paper_cost_l3526_352610


namespace equation_is_linear_l3526_352653

/-- A linear equation with two variables is of the form ax + by = c, where a and b are not both zero -/
def is_linear_equation_two_vars (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x + y = 2 -/
def equation (x y : ℝ) : Prop := x + y = 2

theorem equation_is_linear : is_linear_equation_two_vars equation :=
sorry

end equation_is_linear_l3526_352653


namespace ellipse_equation_l3526_352648

/-- Given an ellipse with the specified properties, prove its equation -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 1/9) -- eccentricity = 1/3
  (F1 F2 P : ℝ × ℝ) -- foci and a point on the ellipse
  (h4 : (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2 = 4 * (a^2 - b^2)) -- distance between foci
  (h5 : (P.1 - F1.1)^2 + (P.2 - F1.2)^2 + 
        (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 4 * a^2) -- sum of distances from foci
  (h6 : ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) / 
        (((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))^(1/2) = 3/5) -- cos∠F1PF2
  (h7 : 1/2 * ((P.1 - F1.1) * (P.2 - F2.2) - (P.2 - F1.2) * (P.1 - F2.1)) = 4) -- area of triangle
  : a^2 = 9 ∧ b^2 = 8 := by
  sorry

end ellipse_equation_l3526_352648


namespace carlos_laundry_time_l3526_352601

/-- The total time for Carlos's laundry process -/
def laundry_time (wash_times : List Nat) (dry_times : List Nat) : Nat :=
  wash_times.sum + dry_times.sum

/-- Theorem stating that Carlos's laundry takes 380 minutes in total -/
theorem carlos_laundry_time :
  laundry_time [30, 45, 40, 50, 35] [85, 95] = 380 := by
  sorry

end carlos_laundry_time_l3526_352601


namespace correct_number_of_pair_sets_l3526_352656

/-- The number of ways to form 6 pairs of balls with different colors -/
def number_of_pair_sets (green red blue : ℕ) : ℕ :=
  if green = 3 ∧ red = 4 ∧ blue = 5 then 1440 else 0

/-- Theorem stating the correct number of pair sets for the given ball counts -/
theorem correct_number_of_pair_sets :
  number_of_pair_sets 3 4 5 = 1440 := by sorry

end correct_number_of_pair_sets_l3526_352656


namespace plane_to_center_distance_l3526_352699

/-- Represents a point on the surface of a sphere -/
structure SpherePoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points on the sphere -/
def distance (p q : SpherePoint) : ℝ := sorry

/-- The radius of the sphere -/
def sphereRadius : ℝ := 13

/-- Theorem: The distance from the plane passing through A, B, C to the sphere center -/
theorem plane_to_center_distance 
  (A B C : SpherePoint) 
  (h1 : distance A B = 6)
  (h2 : distance B C = 8)
  (h3 : distance C A = 10) :
  ∃ (d : ℝ), d = 12 ∧ d^2 + sphereRadius^2 = (distance A B)^2 + (distance B C)^2 + (distance C A)^2 := by
  sorry

end plane_to_center_distance_l3526_352699


namespace cafeteria_extra_apples_l3526_352600

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 32 extra apples -/
theorem cafeteria_extra_apples :
  extra_apples 25 17 10 = 32 := by
  sorry

end cafeteria_extra_apples_l3526_352600


namespace polynomial_descending_order_x_l3526_352698

theorem polynomial_descending_order_x (x y : ℝ) :
  3 * x * y^2 - 2 * x^2 * y - x^3 * y^3 - 4 =
  -x^3 * y^3 - 2 * x^2 * y + 3 * x * y^2 - 4 :=
by sorry

end polynomial_descending_order_x_l3526_352698


namespace shaded_shapes_area_l3526_352638

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a shape on the grid -/
structure GridShape where
  vertices : List GridPoint

/-- The grid size -/
def gridSize : ℕ := 7

/-- Function to calculate the area of a shape on the grid -/
def calculateArea (shape : GridShape) : ℚ :=
  sorry

/-- The newly designed shaded shapes on the grid -/
def shadedShapes : List GridShape :=
  sorry

/-- Theorem stating that the total area of the shaded shapes is 3 -/
theorem shaded_shapes_area :
  (shadedShapes.map calculateArea).sum = 3 := by
  sorry

end shaded_shapes_area_l3526_352638


namespace rational_numbers_include_integers_and_fractions_l3526_352641

/-- A rational number is a number that can be expressed as the quotient of two integers, where the denominator is non-zero. -/
def IsRational (x : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- An integer is a whole number (positive, negative, or zero) without a fractional component. -/
def IsInteger (x : ℚ) : Prop := ∃ (n : ℤ), x = n

/-- A fraction is a rational number that is not an integer. -/
def IsFraction (x : ℚ) : Prop := IsRational x ∧ ¬IsInteger x

theorem rational_numbers_include_integers_and_fractions :
  (∀ x : ℚ, IsInteger x → IsRational x) ∧
  (∀ x : ℚ, IsFraction x → IsRational x) :=
sorry

end rational_numbers_include_integers_and_fractions_l3526_352641


namespace batsman_average_theorem_l3526_352681

/-- Represents a batsman's cricket performance -/
structure BatsmanStats where
  totalInnings : ℕ
  lastInningsScore : ℕ
  averageIncrease : ℚ
  notOutInnings : ℕ

/-- Calculates the average score of a batsman after their latest innings,
    considering 'not out' innings -/
def calculateAdjustedAverage (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.totalInnings * (stats.averageIncrease + 
    (stats.lastInningsScore / stats.totalInnings : ℚ))
  totalRuns / (stats.totalInnings - stats.notOutInnings : ℚ)

/-- Theorem stating that for a batsman with given statistics, 
    their adjusted average is approximately 88.64 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.totalInnings = 25)
  (h2 : stats.lastInningsScore = 150)
  (h3 : stats.averageIncrease = 3)
  (h4 : stats.notOutInnings = 3) :
  abs (calculateAdjustedAverage stats - 88.64) < 0.01 := by
  sorry

end batsman_average_theorem_l3526_352681


namespace gcd_1343_816_l3526_352693

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end gcd_1343_816_l3526_352693


namespace quadratic_inequality_solution_l3526_352684

theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  let solution := {x : ℝ | a * x^2 - (a - 1) * x - 1 < 0}
  ((-1 < a ∧ a < 0) → solution = {x | x < 1 ∨ x > -1/a}) ∧
  (a = -1 → solution = {x | x ≠ 1}) ∧
  (a < -1 → solution = {x | x < -1/a ∨ x > 1}) :=
by sorry

end quadratic_inequality_solution_l3526_352684


namespace clock_hands_angle_at_8_30_clock_hands_angle_at_8_30_is_75_l3526_352665

/-- The angle between clock hands at 8:30 -/
theorem clock_hands_angle_at_8_30 : ℝ :=
  let hours : ℝ := 8
  let minutes : ℝ := 30
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_hand_angle : ℝ := hours * degrees_per_hour + (minutes / 60) * degrees_per_hour
  let minute_hand_angle : ℝ := minutes * degrees_per_minute
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  75

/-- The theorem stating that the angle between clock hands at 8:30 is 75 degrees -/
theorem clock_hands_angle_at_8_30_is_75 : clock_hands_angle_at_8_30 = 75 := by
  sorry

end clock_hands_angle_at_8_30_clock_hands_angle_at_8_30_is_75_l3526_352665


namespace third_month_sales_l3526_352679

def sales_1 : ℕ := 3435
def sales_2 : ℕ := 3927
def sales_4 : ℕ := 4230
def sales_5 : ℕ := 3562
def sales_6 : ℕ := 1991
def target_average : ℕ := 3500
def num_months : ℕ := 6

theorem third_month_sales :
  sales_1 + sales_2 + sales_4 + sales_5 + sales_6 + 3855 = target_average * num_months :=
by sorry

end third_month_sales_l3526_352679


namespace item_list_price_l3526_352669

theorem item_list_price : ∃ (list_price : ℝ), 
  list_price > 0 ∧
  0.15 * (list_price - 15) = 0.25 * (list_price - 25) ∧
  list_price = 40 := by
sorry

end item_list_price_l3526_352669


namespace sin_cos_inverse_equation_l3526_352661

theorem sin_cos_inverse_equation (t : ℝ) :
  (Real.sin (2 * t) - Real.arcsin (2 * t))^2 + (Real.arccos (2 * t) - Real.cos (2 * t))^2 = 1 ↔
  ∃ k : ℤ, t = (π / 8) * (2 * ↑k + 1) :=
by sorry

end sin_cos_inverse_equation_l3526_352661


namespace no_integer_solutions_l3526_352639

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^2 + 3*x*y - 2*y^2 = 122 := by
  sorry

end no_integer_solutions_l3526_352639


namespace sum_and_difference_squares_l3526_352621

theorem sum_and_difference_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) :
  (x + y)^2 + (x - y)^2 = 640 := by
  sorry

end sum_and_difference_squares_l3526_352621


namespace smallest_n_with_all_digit_sums_l3526_352646

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define a function to get all divisors of a number
def divisors (n : ℕ) : Set ℕ := sorry

-- Define a function to get the set of sums of digits of all divisors
def sumsOfDigitsOfDivisors (n : ℕ) : Set ℕ := sorry

-- Main theorem
theorem smallest_n_with_all_digit_sums :
  ∀ n : ℕ, n < 288 →
    ¬(∀ k : ℕ, k ∈ Finset.range 9 → (k + 1) ∈ sumsOfDigitsOfDivisors n) ∧
  (∀ k : ℕ, k ∈ Finset.range 9 → (k + 1) ∈ sumsOfDigitsOfDivisors 288) := by
  sorry

end smallest_n_with_all_digit_sums_l3526_352646


namespace no_solutions_diophantine_equation_l3526_352686

theorem no_solutions_diophantine_equation :
  ¬∃ (n x y k : ℕ), n ≥ 1 ∧ x > 0 ∧ y > 0 ∧ k > 1 ∧ 
  Nat.gcd x y = 1 ∧ 3^n = x^k + y^k :=
sorry

end no_solutions_diophantine_equation_l3526_352686


namespace total_stars_is_580_l3526_352613

/-- The number of stars needed to fill all bottles Kyle bought -/
def total_stars : ℕ :=
  let type_a_initial := 3
  let type_a_later := 5
  let type_b := 4
  let type_c := 2
  let capacity_a := 30
  let capacity_b := 50
  let capacity_c := 70
  (type_a_initial + type_a_later) * capacity_a + type_b * capacity_b + type_c * capacity_c

theorem total_stars_is_580 : total_stars = 580 := by
  sorry

end total_stars_is_580_l3526_352613


namespace determinant_equals_r_plus_s_minus_t_l3526_352688

def quartic_polynomial (r s t : ℝ) (x : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

def det_matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![1+a, 1, 1, 1],
    ![1, 1+b, 1, 1],
    ![1, 1, 1+c, 1],
    ![1, 1, 1, 1+d]]

theorem determinant_equals_r_plus_s_minus_t (r s t : ℝ) (a b c d : ℝ) :
  quartic_polynomial r s t a = 0 →
  quartic_polynomial r s t b = 0 →
  quartic_polynomial r s t c = 0 →
  quartic_polynomial r s t d = 0 →
  Matrix.det (det_matrix a b c d) = r + s - t :=
by sorry

end determinant_equals_r_plus_s_minus_t_l3526_352688


namespace fidos_yard_area_fraction_l3526_352628

theorem fidos_yard_area_fraction :
  let square_side : ℝ := 2  -- Arbitrary side length
  let circle_radius : ℝ := 1  -- Half of the square side
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area / square_area = π :=
by sorry

end fidos_yard_area_fraction_l3526_352628


namespace seating_arrangement_solution_l3526_352634

/-- Represents a seating arrangement with rows of 7 or 9 seats. -/
structure SeatingArrangement where
  rows_of_nine : ℕ
  rows_of_seven : ℕ

/-- 
  Theorem: Given a seating arrangement where each row seats either 7 or 9 people, 
  and 61 people are to be seated with every seat occupied, 
  the number of rows seating exactly 9 people is 6.
-/
theorem seating_arrangement_solution : 
  ∃ (arrangement : SeatingArrangement),
    arrangement.rows_of_nine * 9 + arrangement.rows_of_seven * 7 = 61 ∧
    arrangement.rows_of_nine = 6 := by
  sorry

end seating_arrangement_solution_l3526_352634


namespace max_reciprocal_sum_l3526_352650

theorem max_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1)
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ z w : ℝ, a^z = 3 → b^w = 3 → 1/z + 1/w ≤ 1) :=
sorry

end max_reciprocal_sum_l3526_352650
