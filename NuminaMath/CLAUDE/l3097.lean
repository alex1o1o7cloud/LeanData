import Mathlib

namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3097_309721

-- Define the real number x
variable (x : ℝ)

-- Define condition p
def p (x : ℝ) : Prop := |x - 2| < 1

-- Define condition q
def q (x : ℝ) : Prop := 1 < x ∧ x < 5

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry


end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3097_309721


namespace NUMINAMATH_CALUDE_min_dwarves_at_risk_l3097_309794

/-- Represents the color of a hat -/
inductive HatColor
| Black
| White

/-- Represents a dwarf with a hat -/
structure Dwarf :=
  (hat : HatColor)

/-- A line of dwarves -/
def DwarfLine := List Dwarf

/-- The probability of guessing correctly for a single dwarf -/
def guessProb : ℚ := 1/2

/-- The minimum number of dwarves at risk given a strategy -/
def minRisk (p : ℕ) (strategy : DwarfLine → ℕ) : ℕ :=
  min p (strategy (List.replicate p (Dwarf.mk HatColor.Black)))

theorem min_dwarves_at_risk (p : ℕ) (h : p > 0) :
  ∃ (strategy : DwarfLine → ℕ), minRisk p strategy = 1 :=
sorry

end NUMINAMATH_CALUDE_min_dwarves_at_risk_l3097_309794


namespace NUMINAMATH_CALUDE_line_equation_l3097_309714

/-- Given a line passing through (-a, 0) and forming a triangle in the second quadrant with area T,
    prove that its equation is 2Tx - a²y + 2aT = 0 -/
theorem line_equation (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = -a ∧ y = 0) ∨ (x < 0 ∧ y > 0) →
    (y = m * x + b ↔ 2 * T * x - a^2 * y + 2 * a * T = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3097_309714


namespace NUMINAMATH_CALUDE_james_score_problem_l3097_309766

theorem james_score_problem (field_goals : ℕ) (shots : ℕ) (total_points : ℕ) 
  (h1 : field_goals = 13)
  (h2 : shots = 20)
  (h3 : total_points = 79) :
  ∃ (points_per_shot : ℕ), 
    field_goals * 3 + shots * points_per_shot = total_points ∧ 
    points_per_shot = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_score_problem_l3097_309766


namespace NUMINAMATH_CALUDE_probability_multiple_of_seven_l3097_309735

/-- The probability of selecting a page number that is a multiple of 7 from a book with 500 pages -/
theorem probability_multiple_of_seven (total_pages : ℕ) (h : total_pages = 500) :
  (Finset.filter (fun n => n % 7 = 0) (Finset.range total_pages)).card / total_pages = 71 / 500 :=
by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_seven_l3097_309735


namespace NUMINAMATH_CALUDE_root_in_interval_l3097_309730

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + x - 5

-- State the theorem
theorem root_in_interval (a b : ℕ+) (x₀ : ℝ) :
  b - a = 1 →
  ∃ x₀, x₀ ∈ Set.Icc a b ∧ f x₀ = 0 →
  a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l3097_309730


namespace NUMINAMATH_CALUDE_b_value_for_decreasing_increasing_cubic_l3097_309765

theorem b_value_for_decreasing_increasing_cubic (a c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -x^3 + a*x^2 + b*x + c
  (∀ x < 0, ∀ y < 0, x < y → f x > f y) →
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f x < f y) →
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_b_value_for_decreasing_increasing_cubic_l3097_309765


namespace NUMINAMATH_CALUDE_james_coin_sale_l3097_309799

/-- Proves the number of coins James needs to sell to recoup his investment -/
theorem james_coin_sale (initial_price : ℝ) (num_coins : ℕ) (price_increase_ratio : ℝ) :
  initial_price = 15 →
  num_coins = 20 →
  price_increase_ratio = 2/3 →
  let total_investment := initial_price * num_coins
  let new_price := initial_price * (1 + price_increase_ratio)
  let coins_to_sell := total_investment / new_price
  ⌊coins_to_sell⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_coin_sale_l3097_309799


namespace NUMINAMATH_CALUDE_engineer_is_smith_l3097_309731

-- Define the cities
inductive City
| Sheffield
| Leeds
| Halfway

-- Define the occupations
inductive Occupation
| Businessman
| Conductor
| Stoker
| Engineer

-- Define the people
structure Person where
  name : String
  occupation : Occupation
  city : City

-- Define the problem setup
def setup : Prop := ∃ (smith robinson jones : Person) 
  (conductor stoker engineer : Person),
  -- Businessmen
  smith.occupation = Occupation.Businessman ∧
  robinson.occupation = Occupation.Businessman ∧
  jones.occupation = Occupation.Businessman ∧
  -- Railroad workers
  conductor.occupation = Occupation.Conductor ∧
  stoker.occupation = Occupation.Stoker ∧
  engineer.occupation = Occupation.Engineer ∧
  -- Locations
  robinson.city = City.Sheffield ∧
  conductor.city = City.Sheffield ∧
  jones.city = City.Leeds ∧
  stoker.city = City.Leeds ∧
  smith.city = City.Halfway ∧
  engineer.city = City.Halfway ∧
  -- Salary relations
  ∃ (conductor_namesake : Person),
    conductor_namesake.name = conductor.name ∧
    conductor_namesake.occupation = Occupation.Businessman ∧
  -- Billiards game
  (∃ (smith_worker : Person),
    smith_worker.name = "Smith" ∧
    smith_worker.occupation ≠ Occupation.Businessman ∧
    smith_worker ≠ stoker) ∧
  -- Engineer's salary relation
  ∃ (closest_businessman : Person),
    closest_businessman.occupation = Occupation.Businessman ∧
    closest_businessman.city = City.Halfway

-- The theorem to prove
theorem engineer_is_smith (h : setup) : 
  ∃ (engineer : Person), engineer.occupation = Occupation.Engineer ∧ 
  engineer.name = "Smith" := by
  sorry

end NUMINAMATH_CALUDE_engineer_is_smith_l3097_309731


namespace NUMINAMATH_CALUDE_batsman_innings_properties_l3097_309790

/-- Represents a cricket batsman's innings statistics -/
structure BatsmanInnings where
  total_runs : ℕ
  total_balls : ℕ
  singles : ℕ
  doubles : ℕ

/-- Calculates the percentage of runs scored by running between wickets -/
def runs_by_running_percentage (innings : BatsmanInnings) : ℚ :=
  ((innings.singles + 2 * innings.doubles : ℚ) / innings.total_runs) * 100

/-- Calculates the strike rate of the batsman -/
def strike_rate (innings : BatsmanInnings) : ℚ :=
  (innings.total_runs : ℚ) / innings.total_balls * 100

/-- Theorem stating the properties of the given batsman's innings -/
theorem batsman_innings_properties :
  let innings : BatsmanInnings := {
    total_runs := 180,
    total_balls := 120,
    singles := 35,
    doubles := 15
  }
  runs_by_running_percentage innings = 36.11 ∧
  strike_rate innings = 150 := by
  sorry

end NUMINAMATH_CALUDE_batsman_innings_properties_l3097_309790


namespace NUMINAMATH_CALUDE_candy_left_after_eating_l3097_309798

/-- The number of candy pieces left after two people eat some from a total collection --/
def candy_left (total : ℕ) (people : ℕ) (eaten_per_person : ℕ) : ℕ :=
  total - (people * eaten_per_person)

/-- Theorem stating that 60 pieces of candy are left when 2 people each eat 4 pieces from a total of 68 --/
theorem candy_left_after_eating : 
  candy_left 68 2 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_candy_left_after_eating_l3097_309798


namespace NUMINAMATH_CALUDE_inequality_proof_l3097_309793

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (2/y) ≥ 25 / (1 + 48*x*y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3097_309793


namespace NUMINAMATH_CALUDE_f_72_value_l3097_309715

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def FunctionF (f : ℕ → ℝ) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a + f b

theorem f_72_value (f : ℕ → ℝ) (p q : ℝ) 
    (h1 : FunctionF f) (h2 : f 2 = q) (h3 : f 3 = p) : 
    f 72 = 2 * p + 3 * q := by
  sorry

end NUMINAMATH_CALUDE_f_72_value_l3097_309715


namespace NUMINAMATH_CALUDE_tan_and_sin_values_l3097_309718

theorem tan_and_sin_values (α : ℝ) (h : Real.tan (α + π / 4) = -3) : 
  Real.tan α = 1 ∧ Real.sin (2 * α + π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_sin_values_l3097_309718


namespace NUMINAMATH_CALUDE_claire_took_eight_photos_l3097_309743

/-- The number of photos taken by Claire -/
def claire_photos : ℕ := 8

/-- The number of photos taken by Lisa -/
def lisa_photos : ℕ := 3 * claire_photos

/-- The number of photos taken by Robert -/
def robert_photos : ℕ := claire_photos + 16

/-- Theorem stating that given the conditions, Claire has taken 8 photos -/
theorem claire_took_eight_photos :
  lisa_photos = robert_photos ∧
  lisa_photos = 3 * claire_photos ∧
  robert_photos = claire_photos + 16 →
  claire_photos = 8 := by
  sorry

end NUMINAMATH_CALUDE_claire_took_eight_photos_l3097_309743


namespace NUMINAMATH_CALUDE_viewing_spot_coordinate_l3097_309796

/-- Given two landmarks and a viewing spot in a park, this theorem proves the coordinate of the viewing spot. -/
theorem viewing_spot_coordinate 
  (landmark1 landmark2 : ℝ) 
  (h1 : landmark1 = 150)
  (h2 : landmark2 = 450)
  (h3 : landmark2 > landmark1) :
  let distance := landmark2 - landmark1
  let viewing_spot := landmark1 + (2/3 * distance)
  viewing_spot = 350 := by
sorry

end NUMINAMATH_CALUDE_viewing_spot_coordinate_l3097_309796


namespace NUMINAMATH_CALUDE_exists_function_with_properties_l3097_309724

theorem exists_function_with_properties : ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1) ∧ 
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_exists_function_with_properties_l3097_309724


namespace NUMINAMATH_CALUDE_investment_split_l3097_309700

/-- Proves that given an initial investment of $1500 split between two banks with annual compound
    interest rates of 4% and 6% respectively, if the total amount after three years is $1755,
    then the initial investment in the bank with 4% interest rate is $476.5625. -/
theorem investment_split (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 1500 ∧ 
  x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1755 →
  x = 476.5625 := by sorry

end NUMINAMATH_CALUDE_investment_split_l3097_309700


namespace NUMINAMATH_CALUDE_wire_cutting_l3097_309782

theorem wire_cutting (total_length piece_length : ℝ) 
  (h1 : total_length = 27.9)
  (h2 : piece_length = 3.1) : 
  ⌊total_length / piece_length⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3097_309782


namespace NUMINAMATH_CALUDE_area_of_fourth_square_l3097_309720

/-- Given two right triangles PQR and PRS with a common hypotenuse PR,
    where the squares of the sides have areas 25, 64, 49, and an unknown value,
    prove that the area of the square on PS is 138 square units. -/
theorem area_of_fourth_square (PQ PR QR RS PS : ℝ) : 
  PQ^2 = 25 → QR^2 = 49 → RS^2 = 64 → 
  PQ^2 + QR^2 = PR^2 → PR^2 + RS^2 = PS^2 →
  PS^2 = 138 := by
  sorry

end NUMINAMATH_CALUDE_area_of_fourth_square_l3097_309720


namespace NUMINAMATH_CALUDE_sophia_age_in_three_years_l3097_309727

/-- Represents the current ages of Jeremy, Sebastian, and Sophia --/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  sophia : ℕ

/-- The sum of their ages in three years is 150 --/
def sum_ages_in_three_years (ages : Ages) : Prop :=
  ages.jeremy + 3 + ages.sebastian + 3 + ages.sophia + 3 = 150

/-- Sebastian is 4 years older than Jeremy --/
def sebastian_older (ages : Ages) : Prop :=
  ages.sebastian = ages.jeremy + 4

/-- Jeremy's current age is 40 --/
def jeremy_age (ages : Ages) : Prop :=
  ages.jeremy = 40

/-- Sophia's age three years from now --/
def sophia_future_age (ages : Ages) : ℕ :=
  ages.sophia + 3

/-- Theorem stating Sophia's age three years from now is 60 --/
theorem sophia_age_in_three_years (ages : Ages) 
  (h1 : sum_ages_in_three_years ages) 
  (h2 : sebastian_older ages) 
  (h3 : jeremy_age ages) : 
  sophia_future_age ages = 60 := by
  sorry

end NUMINAMATH_CALUDE_sophia_age_in_three_years_l3097_309727


namespace NUMINAMATH_CALUDE_power_base_property_l3097_309739

theorem power_base_property (k : ℝ) (h : k > 1) :
  let x := k^(1/(k-1))
  ∀ y : ℝ, (k*x)^(y/k) = x^y := by
sorry

end NUMINAMATH_CALUDE_power_base_property_l3097_309739


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3097_309741

theorem cube_sum_theorem (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (prod_sum_condition : x*y + y*z + z*x = 6)
  (prod_condition : x*y*z = -15) :
  x^3 + y^3 + z^3 = -97 := by sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3097_309741


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3097_309797

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (2022 - a - b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3097_309797


namespace NUMINAMATH_CALUDE_power_sum_of_i_l3097_309723

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^22 + i^222 = -2 := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l3097_309723


namespace NUMINAMATH_CALUDE_set_operations_l3097_309761

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}

theorem set_operations :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 2 < x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3097_309761


namespace NUMINAMATH_CALUDE_range_of_m_l3097_309764

theorem range_of_m (m : ℝ) : 
  (¬(∃ x : ℝ, m * x^2 + 1 ≤ 0) ∨ ¬(∀ x : ℝ, x^2 + m * x + 1 > 0)) → m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3097_309764


namespace NUMINAMATH_CALUDE_tape_shortage_l3097_309757

/-- The amount of tape Joy has -/
def tape_amount : ℕ := 180

/-- The width of the field -/
def field_width : ℕ := 35

/-- The length of the field -/
def field_length : ℕ := 80

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem tape_shortage : 
  rectangle_perimeter field_length field_width = tape_amount + 50 := by
  sorry

end NUMINAMATH_CALUDE_tape_shortage_l3097_309757


namespace NUMINAMATH_CALUDE_total_amount_calculation_l3097_309703

/-- Calculates the total amount after simple interest is applied -/
def total_amount_after_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Theorem: The total amount after interest for the given conditions -/
theorem total_amount_calculation :
  let principal : ℝ := 979.0209790209791
  let rate : ℝ := 0.06
  let time : ℝ := 2.4
  total_amount_after_interest principal rate time = 1120.0649350649352 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l3097_309703


namespace NUMINAMATH_CALUDE_approximate_and_scientific_notation_l3097_309756

/-- Determines the place value of the last non-zero digit in a number -/
def lastNonZeroDigitPlace (n : ℕ) : ℕ := sorry

/-- Converts a natural number to scientific notation -/
def toScientificNotation (n : ℕ) : ℝ × ℤ := sorry

theorem approximate_and_scientific_notation :
  (lastNonZeroDigitPlace 24000 = 100) ∧
  (toScientificNotation 46400000 = (4.64, 7)) := by sorry

end NUMINAMATH_CALUDE_approximate_and_scientific_notation_l3097_309756


namespace NUMINAMATH_CALUDE_battery_price_l3097_309702

theorem battery_price (total_cost tire_cost : ℕ) (h1 : total_cost = 224) (h2 : tire_cost = 42) :
  total_cost - 4 * tire_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_battery_price_l3097_309702


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3097_309772

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑a - Complex.I) * (1 + Complex.I) = Complex.I * (Complex.ofReal (a - 1)) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3097_309772


namespace NUMINAMATH_CALUDE_multiplication_value_l3097_309709

theorem multiplication_value : ∃ x : ℚ, (5 / 6) * x = 10 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_l3097_309709


namespace NUMINAMATH_CALUDE_wage_change_equation_l3097_309745

/-- Represents the number of employees in each education category -/
structure EmployeeCount where
  illiterate : ℕ := 20
  primary : ℕ
  college : ℕ

/-- Represents the daily wages before and after the change -/
structure DailyWages where
  illiterate : (ℕ × ℕ) := (25, 10)
  primary : (ℕ × ℕ) := (40, 25)
  college : (ℕ × ℕ) := (50, 60)

/-- The main theorem stating the relationship between employee counts and total employees -/
theorem wage_change_equation (N : ℕ) (emp : EmployeeCount) (wages : DailyWages) :
  N = emp.illiterate + emp.primary + emp.college →
  15 * emp.primary - 10 * emp.college = 10 * N - 300 := by
  sorry

#check wage_change_equation

end NUMINAMATH_CALUDE_wage_change_equation_l3097_309745


namespace NUMINAMATH_CALUDE_barons_claim_correct_l3097_309780

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two different 10-digit numbers satisfying the Baron's claim -/
theorem barons_claim_correct : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    10^9 ≤ a ∧ a < 10^10 ∧
    10^9 ≤ b ∧ b < 10^10 ∧
    a % 10 ≠ 0 ∧
    b % 10 ≠ 0 ∧
    a + sum_of_digits (a^2) = b + sum_of_digits (b^2) :=
by sorry

end NUMINAMATH_CALUDE_barons_claim_correct_l3097_309780


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_6_sqrt_3_l3097_309752

theorem sqrt_sum_equals_6_sqrt_3 :
  Real.sqrt (31 - 12 * Real.sqrt 3) + Real.sqrt (31 + 12 * Real.sqrt 3) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_6_sqrt_3_l3097_309752


namespace NUMINAMATH_CALUDE_least_positive_difference_l3097_309792

def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_sequence (b₁ d : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1) * d

def sequence_A (n : ℕ) : ℝ := geometric_sequence 3 2 n

def sequence_B (n : ℕ) : ℝ := arithmetic_sequence 15 15 n

def valid_term_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_term_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_positive_difference :
  ∃ (m n : ℕ), 
    valid_term_A m ∧ 
    valid_term_B n ∧ 
    (∀ (i j : ℕ), valid_term_A i → valid_term_B j → 
      |sequence_A i - sequence_B j| ≥ |sequence_A m - sequence_B n|) ∧
    |sequence_A m - sequence_B n| = 3 :=
sorry

end NUMINAMATH_CALUDE_least_positive_difference_l3097_309792


namespace NUMINAMATH_CALUDE_gcf_of_26_and_16_l3097_309755

theorem gcf_of_26_and_16 :
  let n : ℕ := 26
  let m : ℕ := 16
  let lcm_nm : ℕ := 52
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_26_and_16_l3097_309755


namespace NUMINAMATH_CALUDE_smallest_product_l3097_309788

def S : Set Int := {-10, -5, 0, 2, 4}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -40 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l3097_309788


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l3097_309770

theorem parallelepiped_dimensions (n : ℕ) : 
  n > 6 →
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) →
  n = 18 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l3097_309770


namespace NUMINAMATH_CALUDE_vector_problem_l3097_309732

def a : Fin 2 → ℝ := ![(-3), 2]
def b : Fin 2 → ℝ := ![2, 1]
def c : Fin 2 → ℝ := ![3, (-1)]

theorem vector_problem (t : ℝ) :
  (∃ (t_min : ℝ), t_min = 4/5 ∧
    (∀ s : ℝ, ‖a + s • b‖ ≥ ‖a + t_min • b‖) ∧
    ‖a + t_min • b‖ = 7 / Real.sqrt 5) ∧
  (∃ (k : ℝ), a - 3/5 • b = k • c) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3097_309732


namespace NUMINAMATH_CALUDE_power_of_two_expression_l3097_309769

theorem power_of_two_expression : 2^4 * 2^2 + 2^4 / 2^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_expression_l3097_309769


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3097_309771

/-- The equation of the tangent line to y = xe^x + 1 at (1, e+1) -/
theorem tangent_line_equation (x y : ℝ) : 
  (∀ t, y = t * Real.exp t + 1) →  -- Curve equation
  2 * Real.exp 1 * x - y - Real.exp 1 + 1 = 0 -- Tangent line equation
  ↔ 
  (x = 1 ∧ y = Real.exp 1 + 1) -- Point of tangency
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3097_309771


namespace NUMINAMATH_CALUDE_sine_function_monotonicity_l3097_309737

theorem sine_function_monotonicity (ω : ℝ) (h1 : ω > 0) : 
  (∀ x ∈ Set.Icc (-π/3) (π/4), 
    ∀ y ∈ Set.Icc (-π/3) (π/4), 
    x < y → 2 * Real.sin (ω * x) < 2 * Real.sin (ω * y)) 
  → 0 < ω ∧ ω ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_monotonicity_l3097_309737


namespace NUMINAMATH_CALUDE_sin_80_gt_sqrt3_sin_10_l3097_309759

theorem sin_80_gt_sqrt3_sin_10 : Real.sin (80 * π / 180) > Real.sqrt 3 * Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_80_gt_sqrt3_sin_10_l3097_309759


namespace NUMINAMATH_CALUDE_prime_equation_solution_l3097_309717

theorem prime_equation_solution (p : ℕ) : 
  (∃ x y : ℕ+, x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p ∈ ({2, 3, 7} : Set ℕ) ∧ Nat.Prime p :=
sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l3097_309717


namespace NUMINAMATH_CALUDE_max_points_in_tournament_l3097_309742

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams.choose 2 * t.games_per_pair

/-- Calculates the total points available in the tournament -/
def total_points (t : Tournament) : ℕ :=
  total_games t * t.points_for_win

/-- Represents the maximum points achievable by top teams -/
def max_points_for_top_teams (t : Tournament) : ℕ :=
  let points_from_top_matches := (t.num_teams - 1) * t.points_for_win
  let points_from_other_matches := (t.num_teams - 3) * 2 * t.points_for_win
  points_from_top_matches + points_from_other_matches

/-- The main theorem to be proved -/
theorem max_points_in_tournament (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_pair = 2)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  max_points_for_top_teams t = 38 :=
sorry

end NUMINAMATH_CALUDE_max_points_in_tournament_l3097_309742


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3097_309787

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the latus rectum of the parabola
def latus_rectum (x : ℝ) : Prop := x = -1

-- Define the length of the line segment
def line_segment_length (b y : ℝ) : Prop := 2 * y = b

-- Main theorem
theorem hyperbola_parabola_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, hyperbola a b x y ∧ parabola x y ∧ latus_rectum x ∧ line_segment_length b y) →
  a = 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3097_309787


namespace NUMINAMATH_CALUDE_subset_partition_with_closure_l3097_309712

theorem subset_partition_with_closure (A B C : Set ℕ+) : 
  (A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅) ∧ 
  (A ∪ B ∪ C = Set.univ) ∧
  (∀ a ∈ A, ∀ b ∈ B, ∀ c ∈ C, (a + c : ℕ+) ∈ A ∧ (b + c : ℕ+) ∈ B ∧ (a + b : ℕ+) ∈ C) →
  ((A = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 2} ∧ 
    B = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 1} ∧ 
    C = {n : ℕ+ | ∃ k : ℕ+, n = 3*k}) ∨
   (A = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 1} ∧ 
    B = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 2} ∧ 
    C = {n : ℕ+ | ∃ k : ℕ+, n = 3*k})) :=
by sorry

end NUMINAMATH_CALUDE_subset_partition_with_closure_l3097_309712


namespace NUMINAMATH_CALUDE_no_real_intersection_l3097_309767

theorem no_real_intersection : ¬∃ (x y : ℝ), y = 8 / (x^3 + 4*x + 3) ∧ x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_intersection_l3097_309767


namespace NUMINAMATH_CALUDE_range_of_m_l3097_309744

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2*x - 8 < 0}

-- Define the universe U as the real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x, x ∈ (U \ M m) ∩ N) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3097_309744


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3097_309773

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^5 - 2*X^3 + X - 1) * (X^3 - X + 1) = (X^2 + X + 1) * q + (2*X) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3097_309773


namespace NUMINAMATH_CALUDE_madeline_and_brother_total_money_l3097_309701

def madeline_money : ℕ := 48

theorem madeline_and_brother_total_money :
  madeline_money + (madeline_money / 2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_madeline_and_brother_total_money_l3097_309701


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l3097_309775

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20460 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l3097_309775


namespace NUMINAMATH_CALUDE_total_pennies_thrown_l3097_309778

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
def pennies_thrown (rachelle gretchen rocky : ℕ) : ℕ := rachelle + gretchen + rocky

/-- Theorem: The total number of pennies thrown is 300 -/
theorem total_pennies_thrown : 
  ∀ (rachelle gretchen rocky : ℕ),
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rocky = gretchen / 3 →
  pennies_thrown rachelle gretchen rocky = 300 := by
sorry

end NUMINAMATH_CALUDE_total_pennies_thrown_l3097_309778


namespace NUMINAMATH_CALUDE_chairs_to_remove_l3097_309704

theorem chairs_to_remove (initial_chairs : Nat) (chairs_per_row : Nat) (participants : Nat) 
  (chairs_to_remove : Nat) :
  initial_chairs = 169 →
  chairs_per_row = 13 →
  participants = 95 →
  chairs_to_remove = 65 →
  (initial_chairs - chairs_to_remove) % chairs_per_row = 0 ∧
  initial_chairs - chairs_to_remove ≥ participants ∧
  ∀ n : Nat, n < chairs_to_remove → 
    (initial_chairs - n) % chairs_per_row ≠ 0 ∨ 
    initial_chairs - n < participants := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l3097_309704


namespace NUMINAMATH_CALUDE_door_diagonal_equation_l3097_309786

theorem door_diagonal_equation (x : ℝ) : x ^ 2 - (x - 2) ^ 2 = (x - 4) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_door_diagonal_equation_l3097_309786


namespace NUMINAMATH_CALUDE_pave_hall_l3097_309705

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℚ :=
  (hall_length * hall_width * 100) / (stone_length * stone_width)

/-- Theorem stating that 2700 stones are required to pave the given hall -/
theorem pave_hall : stones_required 36 15 4 5 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_pave_hall_l3097_309705


namespace NUMINAMATH_CALUDE_checkerboard_coverage_uncoverable_boards_l3097_309751

/-- Represents a rectangular checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Checks if a checkerboard can be completely covered by non-overlapping dominos -/
def can_cover (board : Checkerboard) : Prop :=
  Even (board.rows * board.cols)

/-- Theorem: A checkerboard can be covered iff its total number of squares is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_cover board ↔ Even (board.rows * board.cols) :=
sorry

/-- Examples of checkerboards -/
def board1 := Checkerboard.mk 4 6
def board2 := Checkerboard.mk 5 5
def board3 := Checkerboard.mk 4 7
def board4 := Checkerboard.mk 5 6
def board5 := Checkerboard.mk 3 7

/-- Theorem: Specific boards that cannot be covered -/
theorem uncoverable_boards :
  ¬(can_cover board2) ∧ ¬(can_cover board5) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_uncoverable_boards_l3097_309751


namespace NUMINAMATH_CALUDE_square_root_16_divided_by_2_l3097_309779

theorem square_root_16_divided_by_2 : Real.sqrt 16 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_16_divided_by_2_l3097_309779


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l3097_309716

theorem solution_satisfies_equations :
  let x : ℤ := -46
  let y : ℤ := -10
  (3 * x - 14 * y = 2) ∧ (4 * y - x = 6) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l3097_309716


namespace NUMINAMATH_CALUDE_function_maximum_ratio_l3097_309747

/-- Given a function f(x) = x³ + ax² + bx - a² - 7a, if f(x) attains a maximum
    value of 10 at x = 1, then b/a = -3/2 -/
theorem function_maximum_ratio (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f x < f 1) ∧ 
  f 1 = 10 → 
  b / a = -3/2 :=
sorry

end NUMINAMATH_CALUDE_function_maximum_ratio_l3097_309747


namespace NUMINAMATH_CALUDE_negation_of_forall_ge_two_l3097_309754

theorem negation_of_forall_ge_two :
  (¬ (∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_ge_two_l3097_309754


namespace NUMINAMATH_CALUDE_product_four_consecutive_even_divisible_by_96_largest_divisor_four_consecutive_even_l3097_309725

/-- The product of four consecutive even natural numbers is always divisible by 96 -/
theorem product_four_consecutive_even_divisible_by_96 :
  ∀ n : ℕ, 96 ∣ (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6) :=
by sorry

/-- 96 is the largest natural number that always divides the product of four consecutive even natural numbers -/
theorem largest_divisor_four_consecutive_even :
  ∀ m : ℕ, (∀ n : ℕ, m ∣ (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6)) → m ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_product_four_consecutive_even_divisible_by_96_largest_divisor_four_consecutive_even_l3097_309725


namespace NUMINAMATH_CALUDE_radar_coverage_theorem_l3097_309738

-- Define constants
def num_radars : ℕ := 8
def radar_radius : ℝ := 15
def ring_width : ℝ := 18

-- Define the theorem
theorem radar_coverage_theorem :
  let center_to_radar : ℝ := 12 / Real.sin (22.5 * π / 180)
  let ring_area : ℝ := 432 * π / Real.tan (22.5 * π / 180)
  (∀ (r : ℝ), r = center_to_radar →
    (num_radars : ℝ) * (2 * radar_radius - ring_width) = 2 * π * r * Real.sin (π / num_radars)) ∧
  (∀ (A : ℝ), A = ring_area →
    A = π * ((r + ring_width / 2)^2 - (r - ring_width / 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_radar_coverage_theorem_l3097_309738


namespace NUMINAMATH_CALUDE_min_value_expression_l3097_309728

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 ∧ 
  (m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) = 6 ↔ m - 2 * n = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3097_309728


namespace NUMINAMATH_CALUDE_outfit_combinations_l3097_309753

/-- The number of different outfits that can be created -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (blazers : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (blazers + 1)

/-- Theorem stating the number of outfits given specific quantities -/
theorem outfit_combinations :
  number_of_outfits 5 4 5 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3097_309753


namespace NUMINAMATH_CALUDE_cricket_team_size_l3097_309784

theorem cricket_team_size :
  ∀ (n : ℕ) (team_avg : ℝ) (keeper_age : ℝ) (remaining_avg : ℝ),
    team_avg = 24 →
    keeper_age = team_avg + 3 →
    remaining_avg = team_avg - 1 →
    (n : ℝ) * team_avg = (n - 2 : ℝ) * remaining_avg + team_avg + keeper_age →
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3097_309784


namespace NUMINAMATH_CALUDE_no_right_triangle_perimeter_area_equality_l3097_309758

theorem no_right_triangle_perimeter_area_equality :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (a + b + Real.sqrt (a^2 + b^2))^2 = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_perimeter_area_equality_l3097_309758


namespace NUMINAMATH_CALUDE_sequences_satisfy_conditions_l3097_309774

-- Define the sequences A and B
def A (n : ℕ) : ℝ × ℝ := (n, n^3)
def B (n : ℕ) : ℝ × ℝ := (-n, -n^3)

-- Define a function to check if a point is on a line through two other points
def is_on_line (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

-- State the theorem
theorem sequences_satisfy_conditions :
  ∀ (i j k : ℕ), 1 ≤ i → i < j → j < k →
    (is_on_line (A i) (A j) (B k) ↔ k = i + j) ∧
    (is_on_line (B i) (B j) (A k) ↔ k = i + j) :=
by sorry

end NUMINAMATH_CALUDE_sequences_satisfy_conditions_l3097_309774


namespace NUMINAMATH_CALUDE_right_triangle_area_twice_hypotenuse_l3097_309733

theorem right_triangle_area_twice_hypotenuse :
  ∃ (a : ℝ), a > 0 ∧
  let hypotenuse := Real.sqrt (2 * a^2)
  let area := a^2 / 2
  (area = 2 * hypotenuse) ∧ (a = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_twice_hypotenuse_l3097_309733


namespace NUMINAMATH_CALUDE_unit_digit_of_fraction_l3097_309795

theorem unit_digit_of_fraction : 
  (998 * 999 * 1000 * 1001 * 1002 * 1003) / 10000 % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_fraction_l3097_309795


namespace NUMINAMATH_CALUDE_workshop_workers_l3097_309734

theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (non_technician_salary : ℕ) : 
  average_salary = 8000 →
  technician_count = 7 →
  technician_salary = 14000 →
  non_technician_salary = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * average_salary = technician_count * technician_salary + (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 28 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3097_309734


namespace NUMINAMATH_CALUDE_max_one_truthful_dwarf_l3097_309706

/-- Represents the height claim of a dwarf -/
structure HeightClaim where
  position : Nat
  claimed_height : Nat

/-- The problem setup for the seven dwarfs -/
def dwarfs_problem : List HeightClaim :=
  [
    ⟨1, 60⟩,
    ⟨2, 61⟩,
    ⟨3, 62⟩,
    ⟨4, 63⟩,
    ⟨5, 64⟩,
    ⟨6, 65⟩,
    ⟨7, 66⟩
  ]

/-- A function to count the maximum number of truthful dwarfs -/
def max_truthful_dwarfs (claims : List HeightClaim) : Nat :=
  sorry

/-- The theorem stating that the maximum number of truthful dwarfs is 1 -/
theorem max_one_truthful_dwarf :
  max_truthful_dwarfs dwarfs_problem = 1 :=
sorry

end NUMINAMATH_CALUDE_max_one_truthful_dwarf_l3097_309706


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3097_309713

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 
  1 / x + 1 / y = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3097_309713


namespace NUMINAMATH_CALUDE_competition_problems_l3097_309719

/-- The total number of problems in the competition. -/
def total_problems : ℕ := 71

/-- The number of problems Lukáš correctly solved. -/
def solved_problems : ℕ := 12

/-- The additional points Lukáš would have gained if he solved the last 12 problems. -/
def additional_points : ℕ := 708

/-- The sum of the first n natural numbers. -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the total number of problems in the competition. -/
theorem competition_problems :
  (sum_first_n solved_problems) +
  (sum_first_n solved_problems + additional_points) =
  sum_first_n total_problems - sum_first_n (total_problems - solved_problems) :=
by sorry

end NUMINAMATH_CALUDE_competition_problems_l3097_309719


namespace NUMINAMATH_CALUDE_square_area_with_corner_circles_l3097_309763

theorem square_area_with_corner_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by sorry

end NUMINAMATH_CALUDE_square_area_with_corner_circles_l3097_309763


namespace NUMINAMATH_CALUDE_num_distinct_paths_l3097_309749

/-- The number of rows in the grid -/
def rows : ℕ := 6

/-- The number of columns in the grid -/
def cols : ℕ := 5

/-- The number of dominoes used -/
def num_dominoes : ℕ := 5

/-- The number of moves to the right required to reach the bottom right corner -/
def moves_right : ℕ := cols - 1

/-- The number of moves down required to reach the bottom right corner -/
def moves_down : ℕ := rows - 1

/-- The total number of moves required to reach the bottom right corner -/
def total_moves : ℕ := moves_right + moves_down

/-- Theorem stating the number of distinct paths from top-left to bottom-right corner -/
theorem num_distinct_paths : (total_moves.choose moves_right) = 126 := by
  sorry

end NUMINAMATH_CALUDE_num_distinct_paths_l3097_309749


namespace NUMINAMATH_CALUDE_triangle_equality_condition_l3097_309776

/-- In a triangle ABC with sides a, b, and c, the equation 
    (b² * c²) / (2 * b * c * cos(A)) = b² + c² - 2 * b * c * cos(A) 
    holds if and only if a = b or a = c. -/
theorem triangle_equality_condition (a b c : ℝ) (A : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * c^2) / (2 * b * c * Real.cos A) = b^2 + c^2 - 2 * b * c * Real.cos A ↔ 
  a = b ∨ a = c := by
sorry

end NUMINAMATH_CALUDE_triangle_equality_condition_l3097_309776


namespace NUMINAMATH_CALUDE_traffic_light_probability_theorem_l3097_309722

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) : ℕ :=
  3 * 5 -- 5 seconds before each color change

/-- Calculates the probability of observing a color change -/
def probabilityOfColorChange (cycle : TrafficLightCycle) (observationInterval : ℕ) : ℚ :=
  (changeObservationWindow cycle : ℚ) / (cycleDuration cycle : ℚ)

theorem traffic_light_probability_theorem (cycle : TrafficLightCycle) 
    (h1 : cycle.green = 50)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 40)
    (h4 : observationInterval = 5) :
    probabilityOfColorChange cycle observationInterval = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_theorem_l3097_309722


namespace NUMINAMATH_CALUDE_solve_equation_l3097_309785

theorem solve_equation : 
  ∃ y : ℝ, ((10 - y)^2 = 4 * y^2) ∧ (y = 10/3 ∨ y = -10) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3097_309785


namespace NUMINAMATH_CALUDE_pig_weight_problem_l3097_309707

theorem pig_weight_problem (x y : ℝ) (h1 : x - y = 72) (h2 : x + y = 348) : x = 210 := by
  sorry

end NUMINAMATH_CALUDE_pig_weight_problem_l3097_309707


namespace NUMINAMATH_CALUDE_g_1993_of_2_equals_65_53_l3097_309781

-- Define the function g
def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

-- Define the recursive function g_n
def g_n : ℕ → ℚ → ℚ
  | 0, x => x
  | 1, x => g (g x)
  | (n+2), x => g (g_n (n+1) x)

-- Theorem statement
theorem g_1993_of_2_equals_65_53 : g_n 1993 2 = 65 / 53 := by
  sorry

end NUMINAMATH_CALUDE_g_1993_of_2_equals_65_53_l3097_309781


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3097_309760

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 5 = 0 ∧ x₂^2 + m*x₂ - 5 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3097_309760


namespace NUMINAMATH_CALUDE_total_games_won_l3097_309777

def team_games_won (games_played : ℕ) (win_percentage : ℚ) : ℕ :=
  ⌊(games_played : ℚ) * win_percentage⌋₊

theorem total_games_won :
  let team_a := team_games_won 150 (35/100)
  let team_b := team_games_won 110 (45/100)
  let team_c := team_games_won 200 (30/100)
  team_a + team_b + team_c = 163 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l3097_309777


namespace NUMINAMATH_CALUDE_round_0_9247_to_hundredth_l3097_309746

def round_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_0_9247_to_hundredth :
  round_to_hundredth 0.9247 = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_round_0_9247_to_hundredth_l3097_309746


namespace NUMINAMATH_CALUDE_power_product_squared_l3097_309710

theorem power_product_squared : (3^5 * 6^5)^2 = 3570467226624 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l3097_309710


namespace NUMINAMATH_CALUDE_systematic_sampling_l3097_309729

theorem systematic_sampling (total_products : Nat) (num_samples : Nat) (sampled_second : Nat) : 
  total_products = 100 → 
  num_samples = 5 → 
  sampled_second = 24 → 
  ∃ (interval : Nat) (position : Nat),
    interval = total_products / num_samples ∧
    position = sampled_second % interval ∧
    (position + 3 * interval = 64) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3097_309729


namespace NUMINAMATH_CALUDE_tangent_lines_to_parabola_l3097_309750

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 9

-- Define the point B
def B : ℝ × ℝ := (-1, 2)

-- Define the two lines
def line1 (x : ℝ) : ℝ := -2*x
def line2 (x : ℝ) : ℝ := 6*x + 8

-- Theorem statement
theorem tangent_lines_to_parabola :
  (∃ x₀ : ℝ, line1 x₀ = parabola x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → line1 x < parabola x) ∧
    line1 (B.1) = B.2) ∧
  (∃ x₀ : ℝ, line2 x₀ = parabola x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → line2 x < parabola x) ∧
    line2 (B.1) = B.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_parabola_l3097_309750


namespace NUMINAMATH_CALUDE_horizontal_line_slope_intercept_product_l3097_309789

/-- Given two distinct points on a horizontal line with y-coordinate 20,
    the product of the slope and y-intercept of the line is 0. -/
theorem horizontal_line_slope_intercept_product (C D : ℝ × ℝ) : 
  C.1 ≠ D.1 → C.2 = 20 → D.2 = 20 → 
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2
  m * b = 0 := by sorry

end NUMINAMATH_CALUDE_horizontal_line_slope_intercept_product_l3097_309789


namespace NUMINAMATH_CALUDE_scientific_notation_of_2720000_l3097_309748

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_2720000 :
  toScientificNotation 2720000 = ScientificNotation.mk 2.72 6 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2720000_l3097_309748


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l3097_309708

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 - 9*x + 20 = 0 → x = 4 ∨ x = 5 → min x (15 - x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l3097_309708


namespace NUMINAMATH_CALUDE_cafe_order_combinations_l3097_309726

-- Define the number of menu items
def menu_items : ℕ := 12

-- Define the number of people ordering
def num_people : ℕ := 3

-- Theorem statement
theorem cafe_order_combinations :
  menu_items ^ num_people = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cafe_order_combinations_l3097_309726


namespace NUMINAMATH_CALUDE_rectangle_equation_l3097_309783

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

theorem rectangle_equation (r : Rectangle) :
  r.length = r.width + 12 →
  r.area = 864 →
  r.width * (r.width + 12) = 864 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_equation_l3097_309783


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l3097_309791

theorem number_of_divisors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l3097_309791


namespace NUMINAMATH_CALUDE_high_speed_rail_distance_scientific_notation_l3097_309768

theorem high_speed_rail_distance_scientific_notation :
  9280000000 = 9.28 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_high_speed_rail_distance_scientific_notation_l3097_309768


namespace NUMINAMATH_CALUDE_rowan_rowing_distance_l3097_309736

-- Define the given constants
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4
def still_water_speed : ℝ := 9.75

-- Define the variables
def current_speed : ℝ := sorry
def distance : ℝ := sorry

-- State the theorem
theorem rowan_rowing_distance :
  downstream_time = distance / (still_water_speed + current_speed) ∧
  upstream_time = distance / (still_water_speed - current_speed) →
  distance = 26 := by sorry

end NUMINAMATH_CALUDE_rowan_rowing_distance_l3097_309736


namespace NUMINAMATH_CALUDE_apple_cost_calculation_apple_cost_proof_l3097_309740

/-- Calculates the total cost of apples for a family after a price increase -/
theorem apple_cost_calculation (original_price : ℝ) (price_increase : ℝ) 
  (family_size : ℕ) (pounds_per_person : ℝ) : ℝ :=
  let new_price := original_price * (1 + price_increase)
  let total_pounds := (family_size : ℝ) * pounds_per_person
  new_price * total_pounds

/-- Proves that the total cost for the given scenario is $16 -/
theorem apple_cost_proof : 
  apple_cost_calculation 1.6 0.25 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_apple_cost_proof_l3097_309740


namespace NUMINAMATH_CALUDE_circle_area_tangent_to_hyperbola_and_xaxis_l3097_309711

/-- A hyperbola in the xy-plane -/
def Hyperbola (x y : ℝ) : Prop := x^2 - 20*y^2 = 24

/-- A circle in the xy-plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- A point is on the x-axis if its y-coordinate is 0 -/
def OnXAxis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A circle is tangent to the hyperbola if there exists a point that satisfies both equations -/
def TangentToHyperbola (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ Hyperbola p.1 p.2

/-- A circle is tangent to the x-axis if there exists a point on the circle that is also on the x-axis -/
def TangentToXAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ OnXAxis p

theorem circle_area_tangent_to_hyperbola_and_xaxis :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    let c := Circle center radius
    TangentToHyperbola c ∧ TangentToXAxis c ∧ π * radius^2 = 504 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tangent_to_hyperbola_and_xaxis_l3097_309711


namespace NUMINAMATH_CALUDE_second_digit_is_seven_l3097_309762

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The second digit of a three-digit number satisfying the given condition is 7 -/
theorem second_digit_is_seven (n : ThreeDigitNumber) :
  100 * n.a + 10 * n.b + n.c - (n.a + n.b + n.c) = 261 → n.b = 7 := by
  sorry

#check second_digit_is_seven

end NUMINAMATH_CALUDE_second_digit_is_seven_l3097_309762
