import Mathlib

namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3992_399206

/-- Calculate the entire surface area of a modified cube -/
theorem modified_cube_surface_area :
  let cube_edge : ℝ := 5
  let large_hole_side : ℝ := 2
  let small_hole_side : ℝ := 0.5
  let original_surface_area : ℝ := 6 * cube_edge^2
  let large_holes_area : ℝ := 6 * large_hole_side^2
  let exposed_inner_area : ℝ := 6 * 4 * large_hole_side^2
  let small_holes_area : ℝ := 6 * 4 * small_hole_side^2
  original_surface_area - large_holes_area + exposed_inner_area - small_holes_area = 228 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3992_399206


namespace NUMINAMATH_CALUDE_additional_days_to_double_earnings_l3992_399207

/-- Represents the number of days John has worked so far -/
def days_worked : ℕ := 10

/-- Represents the amount of money John has earned so far in dollars -/
def current_earnings : ℕ := 250

/-- Calculates John's daily rate in dollars -/
def daily_rate : ℚ := current_earnings / days_worked

/-- Calculates the total amount John needs to earn to double his current earnings -/
def target_earnings : ℕ := 2 * current_earnings

/-- Calculates the additional amount John needs to earn -/
def additional_earnings : ℕ := target_earnings - current_earnings

/-- Theorem stating the number of additional days John needs to work -/
theorem additional_days_to_double_earnings : 
  (additional_earnings : ℚ) / daily_rate = 10 := by sorry

end NUMINAMATH_CALUDE_additional_days_to_double_earnings_l3992_399207


namespace NUMINAMATH_CALUDE_pen_price_ratio_l3992_399290

theorem pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
    x > 0 → y > 0 → b > 0 → g > 0 →
    (x + y) * g = 4 * (x * b + y * g) →
    (x + y) * b = (1 / 2) * (x * b + y * g) →
    g = 8 * b := by
  sorry

end NUMINAMATH_CALUDE_pen_price_ratio_l3992_399290


namespace NUMINAMATH_CALUDE_square_existence_l3992_399255

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a square
structure Square where
  side1 : Line2D
  side2 : Line2D
  side3 : Line2D
  side4 : Line2D

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

-- Theorem statement
theorem square_existence
  (A B C D : Point2D)
  (h_not_collinear : ¬(areCollinear A B C ∨ areCollinear A B D ∨ areCollinear A C D ∨ areCollinear B C D)) :
  ∃ (s : Square),
    pointOnLine A s.side1 ∧
    pointOnLine B s.side2 ∧
    pointOnLine C s.side3 ∧
    pointOnLine D s.side4 :=
sorry

end NUMINAMATH_CALUDE_square_existence_l3992_399255


namespace NUMINAMATH_CALUDE_max_equal_distribution_l3992_399263

theorem max_equal_distribution (bags : Nat) (eyeliners : Nat) : 
  bags = 2923 → eyeliners = 3239 → Nat.gcd bags eyeliners = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_equal_distribution_l3992_399263


namespace NUMINAMATH_CALUDE_min_sum_squares_l3992_399200

-- Define the set of possible values
def S : Finset Int := {-6, -4, -1, 0, 3, 5, 7, 10}

-- Define the theorem
theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3992_399200


namespace NUMINAMATH_CALUDE_smallest_odd_six_digit_divisible_by_125_l3992_399237

def is_odd_digit (d : Nat) : Prop := d % 2 = 1 ∧ d < 10

def all_digits_odd (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_odd_digit d

def is_six_digit (n : Nat) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

theorem smallest_odd_six_digit_divisible_by_125 :
  ∀ n : Nat, is_six_digit n → all_digits_odd n → n % 125 = 0 →
  111375 ≤ n := by sorry

end NUMINAMATH_CALUDE_smallest_odd_six_digit_divisible_by_125_l3992_399237


namespace NUMINAMATH_CALUDE_bcd4_hex_to_dec_l3992_399241

def hex_to_dec (digit : Char) : ℕ :=
  match digit with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | d   => d.toString.toNat!

def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun d acc => hex_to_dec d + 16 * acc) 0

theorem bcd4_hex_to_dec :
  hex_string_to_dec "BCD4" = 31444 := by
  sorry

end NUMINAMATH_CALUDE_bcd4_hex_to_dec_l3992_399241


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3992_399213

theorem sixth_term_of_geometric_sequence (a₁ a₉ : ℝ) (h₁ : a₁ = 12) (h₂ : a₉ = 31104) :
  let r := (a₉ / a₁) ^ (1 / 8)
  let a₆ := a₁ * r ^ 5
  a₆ = 93312 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3992_399213


namespace NUMINAMATH_CALUDE_salt_solution_replacement_l3992_399261

/-- Given two solutions with different salt concentrations, prove the fraction of
    the first solution replaced to achieve a specific final concentration -/
theorem salt_solution_replacement
  (initial_salt_concentration : Real)
  (second_salt_concentration : Real)
  (final_salt_concentration : Real)
  (h1 : initial_salt_concentration = 0.14)
  (h2 : second_salt_concentration = 0.22)
  (h3 : final_salt_concentration = 0.16) :
  ∃ (x : Real), 
    x = 1/4 ∧ 
    initial_salt_concentration + x * second_salt_concentration - 
      x * initial_salt_concentration = final_salt_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_replacement_l3992_399261


namespace NUMINAMATH_CALUDE_probability_N_16_mod_7_eq_1_l3992_399228

theorem probability_N_16_mod_7_eq_1 (N : ℕ) : 
  (∃ (k : ℕ), N = k ∧ 1 ≤ k ∧ k ≤ 2027) →
  (Nat.card {k : ℕ | 1 ≤ k ∧ k ≤ 2027 ∧ (k^16 % 7 = 1)}) / 2027 = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_N_16_mod_7_eq_1_l3992_399228


namespace NUMINAMATH_CALUDE_sphere_properties_l3992_399260

/-- For a sphere with volume 72π cubic inches, prove its surface area and diameter -/
theorem sphere_properties (V : ℝ) (h : V = 72 * Real.pi) :
  let r := (3 * V / (4 * Real.pi)) ^ (1/3)
  (4 * Real.pi * r^2 = 36 * Real.pi * 2^(2/3)) ∧
  (2 * r = 6 * 2^(1/3)) := by
sorry

end NUMINAMATH_CALUDE_sphere_properties_l3992_399260


namespace NUMINAMATH_CALUDE_complex_magnitude_l3992_399224

theorem complex_magnitude (z : ℂ) (h : z * (1 + 2*Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3992_399224


namespace NUMINAMATH_CALUDE_ginger_cakes_l3992_399203

/-- The number of cakes Ginger bakes in 10 years --/
def cakes_in_ten_years : ℕ :=
  let children := 2
  let children_holidays := 4
  let husband_holidays := 6
  let parents := 2
  let years := 10
  let cakes_per_year := children * children_holidays + husband_holidays + parents
  cakes_per_year * years

theorem ginger_cakes : cakes_in_ten_years = 160 := by
  sorry

end NUMINAMATH_CALUDE_ginger_cakes_l3992_399203


namespace NUMINAMATH_CALUDE_difference_of_squares_l3992_399292

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3992_399292


namespace NUMINAMATH_CALUDE_complex_sum_equality_l3992_399240

theorem complex_sum_equality (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l3992_399240


namespace NUMINAMATH_CALUDE_purple_socks_added_theorem_l3992_399250

/-- Represents the number of socks of each color -/
structure SockDrawer where
  green : Nat
  purple : Nat
  orange : Nat

/-- The initial state of the sock drawer -/
def initialDrawer : SockDrawer :=
  { green := 6, purple := 18, orange := 12 }

/-- Calculates the total number of socks in a drawer -/
def totalSocks (drawer : SockDrawer) : Nat :=
  drawer.green + drawer.purple + drawer.orange

/-- Calculates the probability of picking a purple sock -/
def purpleProbability (drawer : SockDrawer) : Rat :=
  drawer.purple / (totalSocks drawer)

/-- Adds purple socks to the drawer -/
def addPurpleSocks (drawer : SockDrawer) (n : Nat) : SockDrawer :=
  { drawer with purple := drawer.purple + n }

theorem purple_socks_added_theorem :
  ∃ n : Nat, purpleProbability (addPurpleSocks initialDrawer n) = 3/5 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_purple_socks_added_theorem_l3992_399250


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l3992_399294

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the rug with three regions -/
structure Rug where
  innerLength : ℝ
  innerWidth : ℝ := 2
  middleWidth : ℝ := 6
  outerWidth : ℝ := 10

/-- Calculates the areas of the three regions of the rug -/
def rugAreas (r : Rug) : Fin 3 → ℝ
  | 0 => rectangleArea ⟨r.innerLength, r.innerWidth⟩
  | 1 => rectangleArea ⟨r.innerLength + 4, r.middleWidth⟩ - rectangleArea ⟨r.innerLength, r.innerWidth⟩
  | 2 => rectangleArea ⟨r.innerLength + 8, r.outerWidth⟩ - rectangleArea ⟨r.innerLength + 4, r.middleWidth⟩

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ (r : Rug), isArithmeticProgression (rugAreas r 0) (rugAreas r 1) (rugAreas r 2) →
  r.innerLength = 4 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l3992_399294


namespace NUMINAMATH_CALUDE_absolute_value_and_sqrt_simplification_l3992_399229

theorem absolute_value_and_sqrt_simplification :
  |-Real.sqrt 3| + Real.sqrt 12 + Real.sqrt 3 * (Real.sqrt 3 - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_sqrt_simplification_l3992_399229


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_exists_l3992_399298

-- Define the set of points
def Points : Set (ℤ × ℤ) :=
  {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)}

-- Define the color type
inductive Color
| red
| blue

-- Define what it means for three points to form a right isosceles triangle
def isRightIsosceles (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x1)^2 + (y3 - y1)^2) ∧
  ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1) = 0)

-- State the theorem
theorem right_isosceles_triangle_exists (f : ℤ × ℤ → Color) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ Points ∧ p2 ∈ Points ∧ p3 ∈ Points ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
  f p1 = f p2 ∧ f p2 = f p3 ∧
  isRightIsosceles p1 p2 p3 := by
  sorry


end NUMINAMATH_CALUDE_right_isosceles_triangle_exists_l3992_399298


namespace NUMINAMATH_CALUDE_complex_multiplication_l3992_399262

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 - i) = 1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3992_399262


namespace NUMINAMATH_CALUDE_barbara_shopping_expense_l3992_399293

theorem barbara_shopping_expense (tuna_packs : ℕ) (tuna_price : ℚ) 
  (water_bottles : ℕ) (water_price : ℚ) (total_spent : ℚ) :
  tuna_packs = 5 →
  tuna_price = 2 →
  water_bottles = 4 →
  water_price = (3/2) →
  total_spent = 56 →
  total_spent - (tuna_packs * tuna_price + water_bottles * water_price) = 40 := by
sorry

end NUMINAMATH_CALUDE_barbara_shopping_expense_l3992_399293


namespace NUMINAMATH_CALUDE_equation_equality_l3992_399288

theorem equation_equality (x y z : ℝ) (h1 : x ≠ y) 
  (h2 : (x^2 - y*z) / (x*(1 - y*z)) = (y^2 - x*z) / (y*(1 - x*z))) : 
  x + y + z = 1/x + 1/y + 1/z := by
sorry

end NUMINAMATH_CALUDE_equation_equality_l3992_399288


namespace NUMINAMATH_CALUDE_village_population_proof_l3992_399245

/-- Proves that given a 20% increase followed by a 20% decrease resulting in 9600,
    the initial population must have been 10000 -/
theorem village_population_proof (initial_population : ℝ) : 
  (initial_population * 1.2 * 0.8 = 9600) → initial_population = 10000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_proof_l3992_399245


namespace NUMINAMATH_CALUDE_f_composition_value_l3992_399212

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.sin x

theorem f_composition_value : f (f (7 * Real.pi / 6)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3992_399212


namespace NUMINAMATH_CALUDE_expression_simplification_l3992_399266

theorem expression_simplification (x y z : ℝ) :
  ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3992_399266


namespace NUMINAMATH_CALUDE_gym_charges_twice_a_month_l3992_399233

/-- Represents a gym's monthly charging system -/
structure Gym where
  members : ℕ
  charge_per_payment : ℕ
  monthly_income : ℕ

/-- Calculates the number of times a gym charges its members per month -/
def charges_per_month (g : Gym) : ℕ :=
  g.monthly_income / (g.members * g.charge_per_payment)

/-- Theorem stating that for the given gym conditions, the number of charges per month is 2 -/
theorem gym_charges_twice_a_month :
  let g : Gym := { members := 300, charge_per_payment := 18, monthly_income := 10800 }
  charges_per_month g = 2 := by
  sorry

end NUMINAMATH_CALUDE_gym_charges_twice_a_month_l3992_399233


namespace NUMINAMATH_CALUDE_recess_time_calculation_l3992_399202

/-- Calculates the total recess time based on the number of each grade received -/
def total_recess_time (normal_recess : ℕ) (extra_a : ℕ) (extra_b : ℕ) (extra_c : ℕ) (minus_d : ℕ) 
  (num_a : ℕ) (num_b : ℕ) (num_c : ℕ) (num_d : ℕ) : ℕ :=
  normal_recess + extra_a * num_a + extra_b * num_b + extra_c * num_c - minus_d * num_d

theorem recess_time_calculation : 
  total_recess_time 20 3 2 1 1 10 12 14 5 = 83 := by
  sorry

end NUMINAMATH_CALUDE_recess_time_calculation_l3992_399202


namespace NUMINAMATH_CALUDE_D_nec_not_suff_A_l3992_399272

-- Define propositions A, B, C, and D
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom C_nec_and_suff_B : (B ↔ C)
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem D_nec_not_suff_A : (A → D) ∧ ¬(D → A) := by
  sorry

end NUMINAMATH_CALUDE_D_nec_not_suff_A_l3992_399272


namespace NUMINAMATH_CALUDE_expression_equals_two_l3992_399296

theorem expression_equals_two (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3992_399296


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3992_399282

/-- A quadratic equation ax^2 - 4x - 2 = 0 has real roots if and only if a ≥ -2 and a ≠ 0 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4*x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3992_399282


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3992_399265

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3992_399265


namespace NUMINAMATH_CALUDE_dart_probability_l3992_399289

/-- The probability of a dart landing in the center square of a regular hexagonal dartboard -/
theorem dart_probability (a : ℝ) (h : a > 0) : 
  let hexagon_side := a
  let square_side := a * Real.sqrt 3 / 2
  let hexagon_area := 3 * Real.sqrt 3 / 2 * a^2
  let square_area := (a * Real.sqrt 3 / 2)^2
  square_area / hexagon_area = 1 / (2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_dart_probability_l3992_399289


namespace NUMINAMATH_CALUDE_fair_compensation_is_two_l3992_399274

/-- Represents the scenario of two merchants selling cows and buying sheep --/
structure MerchantScenario where
  num_cows : ℕ
  num_sheep : ℕ
  lamb_price : ℕ

/-- The conditions of the problem --/
def scenario_conditions (s : MerchantScenario) : Prop :=
  ∃ (q : ℕ),
    s.num_sheep = 2 * q + 1 ∧
    s.num_cows ^ 2 = 10 * s.num_sheep + s.lamb_price ∧
    s.lamb_price < 10 ∧
    s.lamb_price > 0

/-- The fair compensation amount --/
def fair_compensation (s : MerchantScenario) : ℕ :=
  (10 - s.lamb_price) / 2

/-- Theorem stating that the fair compensation is 2 yuan --/
theorem fair_compensation_is_two (s : MerchantScenario) 
  (h : scenario_conditions s) : fair_compensation s = 2 := by
  sorry


end NUMINAMATH_CALUDE_fair_compensation_is_two_l3992_399274


namespace NUMINAMATH_CALUDE_thursday_is_only_valid_start_day_l3992_399221

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def DayOfWeek.next (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

def DayOfWeek.addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => (d.addDays n).next

def isOpen (d : DayOfWeek) : Bool :=
  match d with
  | .Sunday => false
  | .Monday => false
  | _ => true

def validRedemptionSchedule (startDay : DayOfWeek) : Bool :=
  let schedule := List.range 8 |>.map (fun i => startDay.addDays (i * 7))
  schedule.all isOpen

theorem thursday_is_only_valid_start_day :
  ∀ (d : DayOfWeek), validRedemptionSchedule d ↔ d = DayOfWeek.Thursday :=
sorry

#check thursday_is_only_valid_start_day

end NUMINAMATH_CALUDE_thursday_is_only_valid_start_day_l3992_399221


namespace NUMINAMATH_CALUDE_absoluteError_2175000_absoluteError_1730000_l3992_399232

/-- Calculates the absolute error of an approximate number -/
def absoluteError (x : ℕ) : ℕ :=
  if x % 10 ≠ 0 then 1
  else if x % 100 ≠ 0 then 10
  else if x % 1000 ≠ 0 then 100
  else if x % 10000 ≠ 0 then 1000
  else 10000

/-- The absolute error of 2175000 is 1 -/
theorem absoluteError_2175000 : absoluteError 2175000 = 1 := by sorry

/-- The absolute error of 1730000 (173 * 10^4) is 10000 -/
theorem absoluteError_1730000 : absoluteError 1730000 = 10000 := by sorry

end NUMINAMATH_CALUDE_absoluteError_2175000_absoluteError_1730000_l3992_399232


namespace NUMINAMATH_CALUDE_max_product_with_constraint_l3992_399286

theorem max_product_with_constraint (a b c : ℕ+) (h : a + 2*b + 3*c = 100) :
  a * b * c ≤ 6171 :=
sorry

end NUMINAMATH_CALUDE_max_product_with_constraint_l3992_399286


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3992_399256

/- Define the constants -/
def total_weight : ℕ := 1000
def round_weight : ℕ := 8
def square_weight : ℕ := 18
def round_price : ℕ := 160
def square_price : ℕ := 270

/- Part 1 -/
theorem part_one (a : ℕ) : 
  round_price * a + square_price * a = 8600 → a = 20 := by sorry

/- Part 2 -/
theorem part_two (x y : ℕ) :
  round_price * x + square_price * y = 16760 ∧
  round_weight * x + square_weight * y = total_weight →
  x = 44 ∧ y = 36 := by sorry

/- Part 3 -/
theorem part_three (m n b : ℕ) :
  b > 0 →
  round_price * m + square_price * n = 16760 ∧
  round_weight * (m + b) + square_weight * n = total_weight →
  (m + b = 80 ∧ n = 20) ∨ (m + b = 116 ∧ n = 4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3992_399256


namespace NUMINAMATH_CALUDE_largest_possible_b_l3992_399244

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (Nat.Prime c) →
  (∀ b' : ℕ, (∃ a' c' : ℕ, 
    (a' * b' * c' = 360) ∧
    (1 < c') ∧
    (c' < b') ∧
    (b' < a') ∧
    (Nat.Prime c')) → b' ≤ b) →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_largest_possible_b_l3992_399244


namespace NUMINAMATH_CALUDE_unique_b_value_l3992_399270

/-- The value of 524123 in base 81 -/
def base_81_value : ℕ := 3 + 2 * 81 + 4 * 81^2 + 1 * 81^3 + 2 * 81^4 + 5 * 81^5

/-- Theorem stating that if b is an integer between 1 and 30 (inclusive),
    and base_81_value - b is divisible by 17, then b must equal 11 -/
theorem unique_b_value (b : ℤ) (h1 : 1 ≤ b) (h2 : b ≤ 30) 
    (h3 : (base_81_value : ℤ) - b ≡ 0 [ZMOD 17]) : b = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l3992_399270


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l3992_399273

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  size : Nat
  h_size : vertices.length = size
  h_size_ge_3 : size ≥ 3

/-- A guard position -/
def Guard := ℝ × ℝ

/-- A point is visible from a guard if the line segment between them doesn't intersect any edge of the polygon -/
def isVisible (p : Polygon) (point guard : ℝ × ℝ) : Prop := sorry

/-- A set of guards covers a polygon if every point in the polygon is visible from at least one guard -/
def covers (p : Polygon) (guards : List Guard) : Prop :=
  ∀ point, ∃ guard ∈ guards, isVisible p point guard

/-- The main theorem: ⌊n/3⌋ guards are sufficient to cover any polygon with n sides -/
theorem art_gallery_theorem (p : Polygon) :
  ∃ guards : List Guard, guards.length ≤ p.size / 3 ∧ covers p guards := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l3992_399273


namespace NUMINAMATH_CALUDE_complement_of_M_l3992_399291

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3992_399291


namespace NUMINAMATH_CALUDE_complement_M_equals_closed_interval_l3992_399218

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

theorem complement_M_equals_closed_interval :
  (Set.univ \ M) = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_M_equals_closed_interval_l3992_399218


namespace NUMINAMATH_CALUDE_modular_sum_equivalence_l3992_399216

theorem modular_sum_equivalence : ∃ (x y z : ℤ), 
  (5 * x) % 29 = 1 ∧ 
  (5 * y) % 29 = 1 ∧ 
  (7 * z) % 29 = 1 ∧ 
  (x + y + z) % 29 = 13 := by
  sorry

end NUMINAMATH_CALUDE_modular_sum_equivalence_l3992_399216


namespace NUMINAMATH_CALUDE_race_time_proof_l3992_399236

/-- 
Given a race with three participants Patrick, Manu, and Amy:
- Patrick's race time is 60 seconds
- Manu's race time is 12 seconds more than Patrick's
- Amy's speed is twice Manu's speed
Prove that Amy's race time is 36 seconds
-/
theorem race_time_proof (patrick_time manu_time amy_time : ℝ) : 
  patrick_time = 60 →
  manu_time = patrick_time + 12 →
  amy_time * 2 = manu_time →
  amy_time = 36 := by
sorry

end NUMINAMATH_CALUDE_race_time_proof_l3992_399236


namespace NUMINAMATH_CALUDE_diagonal_length_l3992_399280

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.B = 12 ∧
  dist q.B q.C = 12 ∧
  dist q.C q.D = 15 ∧
  dist q.D q.A = 15 ∧
  let angle := λ p₁ p₂ p₃ : ℝ × ℝ => Real.arccos (
    ((p₁.1 - p₂.1) * (p₃.1 - p₂.1) + (p₁.2 - p₂.2) * (p₃.2 - p₂.2)) /
    (dist p₁ p₂ * dist p₂ p₃)
  )
  angle q.A q.D q.C = 2 * Real.pi / 3

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_special_quadrilateral q) :
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.C = 15 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l3992_399280


namespace NUMINAMATH_CALUDE_x_expression_l3992_399258

theorem x_expression (m n x : ℝ) (h1 : m ≠ n) (h2 : m ≠ 0) (h3 : n ≠ 0) 
  (h4 : (x + 2*m)^2 - 2*(x + n)^2 = 2*(m - n)^2) : 
  x = 2*m - 2*n := by
sorry

end NUMINAMATH_CALUDE_x_expression_l3992_399258


namespace NUMINAMATH_CALUDE_jerry_bacon_calories_l3992_399231

/-- Represents Jerry's breakfast -/
structure Breakfast where
  pancakes : ℕ
  pancake_calories : ℕ
  bacon_strips : ℕ
  cereal_calories : ℕ
  total_calories : ℕ

/-- Calculates the calories per strip of bacon -/
def bacon_calories_per_strip (b : Breakfast) : ℕ :=
  (b.total_calories - (b.pancakes * b.pancake_calories + b.cereal_calories)) / b.bacon_strips

/-- Theorem stating that each strip of bacon in Jerry's breakfast has 100 calories -/
theorem jerry_bacon_calories :
  let jerry_breakfast : Breakfast := {
    pancakes := 6,
    pancake_calories := 120,
    bacon_strips := 2,
    cereal_calories := 200,
    total_calories := 1120
  }
  bacon_calories_per_strip jerry_breakfast = 100 := by
  sorry

end NUMINAMATH_CALUDE_jerry_bacon_calories_l3992_399231


namespace NUMINAMATH_CALUDE_coin_identification_possible_l3992_399243

/-- Represents the weight of a coin -/
inductive CoinWeight
| Counterfeit
| Genuine

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftLighter
| RightLighter
| Equal

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (weight : CoinWeight)

/-- Represents a weighing on the balance scale -/
def weighing (left : List Coin) (right : List Coin) : WeighingResult :=
  sorry

/-- Represents the set of all coins -/
def allCoins : List Coin :=
  sorry

/-- The number of coins -/
def numCoins : Nat := 14

/-- The number of counterfeit coins -/
def numCounterfeit : Nat := 7

/-- The number of genuine coins -/
def numGenuine : Nat := 7

/-- The maximum number of allowed weighings -/
def maxWeighings : Nat := 3

theorem coin_identification_possible :
  ∃ (strategy : List (List Coin × List Coin)),
    (strategy.length ≤ maxWeighings) ∧
    (∀ (c : Coin), c ∈ allCoins →
      (c.weight = CoinWeight.Counterfeit ↔ c.id ≤ numCounterfeit) ∧
      (c.weight = CoinWeight.Genuine ↔ c.id > numCounterfeit)) :=
  sorry

end NUMINAMATH_CALUDE_coin_identification_possible_l3992_399243


namespace NUMINAMATH_CALUDE_curve_intersects_median_l3992_399211

/-- Given non-collinear points A, B, C in the complex plane corresponding to 
    z₀ = ai, z₁ = 1/2 + bi, z₂ = 1 + ci respectively, where a, b, c are real numbers,
    prove that the curve z = z₀cos⁴t + 2z₁cos²tsin²t + z₂sin⁴t intersects the median 
    of triangle ABC parallel to AC at exactly one point (1/2, (a+c+2b)/4). -/
theorem curve_intersects_median (a b c : ℝ) 
  (h_non_collinear : a + c - 2*b ≠ 0) : 
  ∃! p : ℂ, 
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
      2 * (1/2 + Complex.I * b) * (Real.cos t)^2 * (Real.sin t)^2 + 
      (1 + Complex.I * c) * (Real.sin t)^4) ∧ 
    p.im = (c - a) * p.re + (3*a + 2*b - c)/4 ∧ 
    p = Complex.mk (1/2) ((a + c + 2*b)/4) := by 
  sorry

end NUMINAMATH_CALUDE_curve_intersects_median_l3992_399211


namespace NUMINAMATH_CALUDE_gangster_undetected_speed_l3992_399205

/-- Represents the speed of a moving object -/
structure Speed :=
  (value : ℝ)

/-- Represents the distance between two points -/
structure Distance :=
  (value : ℝ)

/-- Represents a moving police officer -/
structure PoliceOfficer :=
  (speed : Speed)
  (spacing : Distance)

/-- Represents a moving gangster -/
structure Gangster :=
  (speed : Speed)

/-- Determines if a gangster is undetected by police officers -/
def is_undetected (g : Gangster) (p : PoliceOfficer) : Prop :=
  (g.speed.value = 2 * p.speed.value) ∨ (g.speed.value = p.speed.value / 2)

/-- Theorem stating the conditions for a gangster to remain undetected -/
theorem gangster_undetected_speed (v : ℝ) (a : ℝ) :
  ∀ (g : Gangster) (p : PoliceOfficer),
  p.speed.value = v →
  p.spacing.value = 9 * a →
  is_undetected g p :=
sorry

end NUMINAMATH_CALUDE_gangster_undetected_speed_l3992_399205


namespace NUMINAMATH_CALUDE_quadratic_bound_l3992_399201

/-- Given a quadratic function f(x) = a x^2 + b x + c, if for any |u| ≤ 10/11 there exists a v 
    such that |u-v| ≤ 1/11 and |f(v)| ≤ 1, then for all x in [-1, 1], |f(x)| ≤ 2. -/
theorem quadratic_bound (a b c : ℝ) : 
  (∀ u : ℝ, |u| ≤ 10/11 → ∃ v : ℝ, |u - v| ≤ 1/11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_bound_l3992_399201


namespace NUMINAMATH_CALUDE_seedling_problem_l3992_399278

theorem seedling_problem (x : ℕ) : 
  (x^2 + 39 = (x + 1)^2 - 50) → (x^2 + 39 = 1975) :=
by
  sorry

#check seedling_problem

end NUMINAMATH_CALUDE_seedling_problem_l3992_399278


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3992_399284

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 5 = 10 ∧
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = 3) →
  a₁ = -2 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3992_399284


namespace NUMINAMATH_CALUDE_rectangle_to_hexagon_side_length_l3992_399276

theorem rectangle_to_hexagon_side_length :
  ∀ (rectangle_length rectangle_width : ℝ) (hexagon_side : ℝ),
    rectangle_length = 24 →
    rectangle_width = 8 →
    (3 * Real.sqrt 3 / 2) * hexagon_side^2 = rectangle_length * rectangle_width →
    hexagon_side = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_hexagon_side_length_l3992_399276


namespace NUMINAMATH_CALUDE_range_of_m_l3992_399210

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

-- Define proposition P
def P (m : ℝ) : Prop := ∀ x₁ x₂, 2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

-- Define proposition Q
def Q (m : ℝ) : Prop := ∀ x, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ≤ 1 ∨ (2 < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3992_399210


namespace NUMINAMATH_CALUDE_trapezoid_area_coefficient_l3992_399271

-- Define the triangle
def triangle_side_1 : ℝ := 15
def triangle_side_2 : ℝ := 39
def triangle_side_3 : ℝ := 36

-- Define the area formula for the trapezoid
def trapezoid_area (γ δ ω : ℝ) : ℝ := γ * ω - δ * ω^2

-- State the theorem
theorem trapezoid_area_coefficient :
  ∃ (γ : ℝ), 
    (trapezoid_area γ (60/169) triangle_side_2 = 0) ∧
    (trapezoid_area γ (60/169) (triangle_side_2/2) = 
      (1/2) * Real.sqrt (
        (triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_1) *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_2) *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_3)
      )) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_coefficient_l3992_399271


namespace NUMINAMATH_CALUDE_brown_mms_in_first_bag_l3992_399251

/-- The number of bags of M&M's. -/
def num_bags : ℕ := 5

/-- The number of brown M&M's in the second bag. -/
def second_bag : ℕ := 12

/-- The number of brown M&M's in the third bag. -/
def third_bag : ℕ := 8

/-- The number of brown M&M's in the fourth bag. -/
def fourth_bag : ℕ := 8

/-- The number of brown M&M's in the fifth bag. -/
def fifth_bag : ℕ := 3

/-- The average number of brown M&M's per bag. -/
def average : ℕ := 8

/-- Theorem stating the number of brown M&M's in the first bag. -/
theorem brown_mms_in_first_bag :
  ∃ (first_bag : ℕ),
    (first_bag + second_bag + third_bag + fourth_bag + fifth_bag) / num_bags = average ∧
    first_bag = 9 := by
  sorry

end NUMINAMATH_CALUDE_brown_mms_in_first_bag_l3992_399251


namespace NUMINAMATH_CALUDE_gym_income_calculation_l3992_399297

/-- Calculates the monthly income of a gym given its bi-monthly charge and number of members. -/
def gym_monthly_income (bi_monthly_charge : ℕ) (num_members : ℕ) : ℕ :=
  2 * bi_monthly_charge * num_members

/-- Proves that a gym charging $18 twice a month with 300 members makes $10,800 per month. -/
theorem gym_income_calculation :
  gym_monthly_income 18 300 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_gym_income_calculation_l3992_399297


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3992_399285

def A : Set ℝ := {x : ℝ | x^2 + x = 0}
def B : Set ℝ := {x : ℝ | x^2 - x = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3992_399285


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l3992_399235

/-- The number of different condiments available -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties -/
def meat_patty_choices : ℕ := 3

/-- The number of choices for buns -/
def bun_choices : ℕ := 2

/-- Theorem: The total number of different hamburger combinations -/
theorem total_hamburger_combinations : 
  (2 ^ num_condiments) * meat_patty_choices * bun_choices = 6144 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l3992_399235


namespace NUMINAMATH_CALUDE_equation_solution_system_of_equations_solution_l3992_399267

-- Problem 1
theorem equation_solution : 
  let x : ℚ := -1
  (2*x + 1) / 6 - (5*x - 1) / 8 = 7 / 12 := by sorry

-- Problem 2
theorem system_of_equations_solution :
  let x : ℚ := 4
  let y : ℚ := 3
  3*x - 2*y = 6 ∧ 2*x + 3*y = 17 := by sorry

end NUMINAMATH_CALUDE_equation_solution_system_of_equations_solution_l3992_399267


namespace NUMINAMATH_CALUDE_fib_120_mod_5_l3992_399225

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the property that the Fibonacci sequence modulo 5 repeats every 20 terms
axiom fib_mod_5_period_20 : ∀ n : ℕ, fib n % 5 = fib (n % 20) % 5

-- Theorem statement
theorem fib_120_mod_5 : fib 120 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_120_mod_5_l3992_399225


namespace NUMINAMATH_CALUDE_maximum_value_inequality_l3992_399248

theorem maximum_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ M : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(a*b^2 + b*c^2 + c*a^2 - 3*a*b*c)) →
  M ≤ 3 / Real.rpow 4 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_maximum_value_inequality_l3992_399248


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l3992_399295

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_2_pow_20_minus_1 :
  unitsDigit ((2 ^ 20) - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l3992_399295


namespace NUMINAMATH_CALUDE_cat_care_cost_is_40_l3992_399223

/-- The cost to care for a cat at Mr. Sean's veterinary clinic -/
def cat_care_cost : ℕ → Prop
| cost => ∃ (dog_cost : ℕ),
  dog_cost = 60 ∧
  20 * dog_cost + 60 * cost = 3600

/-- Theorem: The cost to care for a cat at Mr. Sean's clinic is $40 -/
theorem cat_care_cost_is_40 : cat_care_cost 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_care_cost_is_40_l3992_399223


namespace NUMINAMATH_CALUDE_store_brand_butter_price_l3992_399259

/-- The price of a single 16 oz package of store-brand butter -/
def single_package_price : ℝ := 6

/-- The price of an 8 oz package of butter -/
def eight_oz_price : ℝ := 4

/-- The normal price of a 4 oz package of butter -/
def four_oz_normal_price : ℝ := 2

/-- The discount rate for 4 oz packages -/
def discount_rate : ℝ := 0.5

/-- The lowest price for 16 oz of butter -/
def lowest_price : ℝ := 6

theorem store_brand_butter_price :
  single_package_price = lowest_price ∧
  lowest_price ≤ eight_oz_price + 2 * (four_oz_normal_price * (1 - discount_rate)) :=
by sorry

end NUMINAMATH_CALUDE_store_brand_butter_price_l3992_399259


namespace NUMINAMATH_CALUDE_equation_solution_l3992_399246

theorem equation_solution : ∃ c : ℚ, (c - 37) / 3 = (3 * c + 7) / 8 ∧ c = -317 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3992_399246


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l3992_399252

theorem power_mod_seventeen : 7^2023 % 17 = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l3992_399252


namespace NUMINAMATH_CALUDE_storybook_pages_l3992_399247

theorem storybook_pages : (10 + 5) / (1 - 1/5 * 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_storybook_pages_l3992_399247


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l3992_399217

theorem quadratic_always_nonnegative (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l3992_399217


namespace NUMINAMATH_CALUDE_final_position_of_942nd_square_l3992_399279

/-- Represents the state of a square after folding -/
structure SquareState where
  position : ℕ
  below : ℕ

/-- Calculates the new state of a square after a fold -/
def fold (state : SquareState) (stripLength : ℕ) : SquareState :=
  if state.position ≤ stripLength then
    state
  else
    { position := 2 * stripLength + 1 - state.position,
      below := stripLength - (2 * stripLength + 1 - state.position) }

/-- Performs multiple folds on a square -/
def foldMultiple (initialState : SquareState) (numFolds : ℕ) : SquareState :=
  match numFolds with
  | 0 => initialState
  | n + 1 => fold (foldMultiple initialState n) (1024 / 2^(n + 1))

theorem final_position_of_942nd_square :
  (foldMultiple { position := 942, below := 0 } 10).below = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_position_of_942nd_square_l3992_399279


namespace NUMINAMATH_CALUDE_mushroom_collectors_problem_l3992_399209

theorem mushroom_collectors_problem :
  ∃ (n m : ℕ),
    n > 0 ∧ m > 0 ∧
    6 + 13 * (n - 1) = 5 + 10 * (m - 1) ∧
    100 < 6 + 13 * (n - 1) ∧
    6 + 13 * (n - 1) < 200 ∧
    n = 14 ∧ m = 18 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_collectors_problem_l3992_399209


namespace NUMINAMATH_CALUDE_joans_spending_l3992_399264

/-- Calculates the total spending on video games after discounts and sales tax --/
def total_spending (basketball_price : ℝ) (basketball_discount : ℝ) 
                   (racing_price : ℝ) (racing_discount : ℝ)
                   (puzzle_price : ℝ) (sales_tax : ℝ) : ℝ :=
  let basketball_discounted := basketball_price * (1 - basketball_discount)
  let racing_discounted := racing_price * (1 - racing_discount)
  let total_before_tax := basketball_discounted + racing_discounted + puzzle_price
  total_before_tax * (1 + sales_tax)

/-- Theorem stating that Joan's total spending on video games is $12.67 --/
theorem joans_spending :
  ∃ (δ : ℝ), δ > 0 ∧ δ < 0.005 ∧ 
  |total_spending 5.20 0.15 4.23 0.10 3.50 0.08 - 12.67| < δ :=
sorry

end NUMINAMATH_CALUDE_joans_spending_l3992_399264


namespace NUMINAMATH_CALUDE_complex_quadratic_equation_solution_l3992_399220

theorem complex_quadratic_equation_solution :
  ∃ (z₁ z₂ : ℂ), 
    (z₁ = 1 + Real.sqrt 3 - (Real.sqrt 3 / 2) * Complex.I) ∧
    (z₂ = 1 - Real.sqrt 3 + (Real.sqrt 3 / 2) * Complex.I) ∧
    (∀ z : ℂ, 3 * z^2 - 2 * z = 7 - 3 * Complex.I ↔ z = z₁ ∨ z = z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_equation_solution_l3992_399220


namespace NUMINAMATH_CALUDE_bag_probability_l3992_399277

theorem bag_probability (n : ℕ) : 
  (6 : ℚ) / (6 + n) = 2 / 5 → n = 9 := by
sorry

end NUMINAMATH_CALUDE_bag_probability_l3992_399277


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3992_399287

/-- A point (x, y) in the coordinate plane that is equally distant from the x-axis, y-axis, 
    line x + 2y = 4, and line y = 2x has x-coordinate equal to -4 / (√5 - 7) -/
theorem equidistant_point_x_coordinate (x y : ℝ) : 
  (abs x = abs y) ∧ 
  (abs x = abs (x + 2*y - 4) / Real.sqrt 5) ∧
  (abs x = abs (y - 2*x) / Real.sqrt 5) →
  x = -4 / (Real.sqrt 5 - 7) := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3992_399287


namespace NUMINAMATH_CALUDE_quadratic_sum_l3992_399219

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (3 * x^2 + 9 * x - 81 = a * (x + b)^2 + c) ∧ (a + b + c = -83.25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3992_399219


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3992_399283

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  (x + 8*y) / (x*y) ≥ 9 :=
sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  (x + 8*y) / (x*y) = 9 ↔ x = 4/3 ∧ y = 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3992_399283


namespace NUMINAMATH_CALUDE_missy_tv_watching_l3992_399253

/-- The number of reality shows Missy watches -/
def num_reality_shows : ℕ := 5

/-- The duration of each reality show in minutes -/
def reality_show_duration : ℕ := 28

/-- The duration of the cartoon in minutes -/
def cartoon_duration : ℕ := 10

/-- The total time Missy spends watching TV in minutes -/
def total_watch_time : ℕ := 150

theorem missy_tv_watching :
  num_reality_shows * reality_show_duration + cartoon_duration = total_watch_time :=
by sorry

end NUMINAMATH_CALUDE_missy_tv_watching_l3992_399253


namespace NUMINAMATH_CALUDE_blue_ball_weight_is_6_l3992_399269

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 9.12 - 3.12

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := 3.12

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := 9.12

theorem blue_ball_weight_is_6 : blue_ball_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_weight_is_6_l3992_399269


namespace NUMINAMATH_CALUDE_largest_divisible_by_9_l3992_399234

def original_number : ℕ := 547654765476

def remove_digits (n : ℕ) (positions : List ℕ) : ℕ :=
  sorry

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem largest_divisible_by_9 :
  ∀ (positions : List ℕ),
    let result := remove_digits original_number positions
    is_divisible_by_9 result →
    result ≤ 5476547646 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_9_l3992_399234


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3992_399299

theorem complex_number_modulus (z : ℂ) : (1 + z) / (1 - z) = Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3992_399299


namespace NUMINAMATH_CALUDE_c_months_equals_six_l3992_399214

def total_cost : ℚ := 435
def a_horses : ℕ := 12
def a_months : ℕ := 8
def b_horses : ℕ := 16
def b_months : ℕ := 9
def c_horses : ℕ := 18
def b_payment : ℚ := 180

theorem c_months_equals_six :
  ∃ (x : ℕ), 
    (b_payment / total_cost) * (a_horses * a_months + b_horses * b_months + c_horses * x) = 
    b_horses * b_months ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_c_months_equals_six_l3992_399214


namespace NUMINAMATH_CALUDE_train_length_is_286_l3992_399254

/-- The speed of the pedestrian in meters per second -/
def pedestrian_speed : ℝ := 1

/-- The speed of the cyclist in meters per second -/
def cyclist_speed : ℝ := 3

/-- The time it takes for the train to pass the pedestrian in seconds -/
def pedestrian_passing_time : ℝ := 22

/-- The time it takes for the train to pass the cyclist in seconds -/
def cyclist_passing_time : ℝ := 26

/-- The speed of the train in meters per second -/
def train_speed : ℝ := 14

/-- The length of the train in meters -/
def train_length : ℝ := (train_speed - pedestrian_speed) * pedestrian_passing_time

theorem train_length_is_286 : train_length = 286 := by
  sorry

end NUMINAMATH_CALUDE_train_length_is_286_l3992_399254


namespace NUMINAMATH_CALUDE_kevins_cards_l3992_399239

/-- Kevin's card problem -/
theorem kevins_cards (initial_cards found_cards : ℕ) 
  (h1 : initial_cards = 65)
  (h2 : found_cards = 539) :
  initial_cards + found_cards = 604 := by
  sorry

end NUMINAMATH_CALUDE_kevins_cards_l3992_399239


namespace NUMINAMATH_CALUDE_polynomial_equality_l3992_399227

/-- Given a polynomial M such that M + (5x^2 - 4x - 3) = -x^2 - 3x,
    prove that M = -6x^2 + x + 3 -/
theorem polynomial_equality (x : ℝ) (M : ℝ → ℝ) : 
  (M x + (5*x^2 - 4*x - 3) = -x^2 - 3*x) → 
  (M x = -6*x^2 + x + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3992_399227


namespace NUMINAMATH_CALUDE_sum_of_squares_l3992_399208

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3992_399208


namespace NUMINAMATH_CALUDE_value_of_b_l3992_399281

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : 
  b = 15 := by sorry

end NUMINAMATH_CALUDE_value_of_b_l3992_399281


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3992_399215

-- Define a line by its slope and y-intercept
def Line (m b : ℝ) := {(x, y) : ℝ × ℝ | y = m * x + b}

-- Define parallel lines
def Parallel (l₁ l₂ : ℝ × ℝ → Prop) :=
  ∃ m b₁ b₂, l₁ = Line m b₁ ∧ l₂ = Line m b₂

theorem parallel_line_equation :
  let l₁ := Line (-4) 1  -- y = -4x + 1
  let l₂ := {(x, y) : ℝ × ℝ | 4 * x + y - 3 = 0}  -- 4x + y - 3 = 0
  Parallel l₁ l₂ ∧ (0, 3) ∈ l₂ := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3992_399215


namespace NUMINAMATH_CALUDE_problem_solution_l3992_399230

theorem problem_solution (a b c : ℕ+) 
  (eq1 : a^3 + 32*b + 2*c = 2018)
  (eq2 : b^3 + 32*a + 2*c = 1115) :
  a^2 + b^2 + c^2 = 226 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3992_399230


namespace NUMINAMATH_CALUDE_smallest_number_in_sample_l3992_399268

/-- Systematic sampling function that returns the smallest number in the sample -/
def systematicSample (totalProducts : ℕ) (sampleSize : ℕ) (containsProduct : ℕ) : ℕ :=
  containsProduct % (totalProducts / sampleSize)

/-- Theorem: The smallest number in the systematic sample is 10 -/
theorem smallest_number_in_sample :
  systematicSample 80 5 42 = 10 := by
  sorry

#eval systematicSample 80 5 42

end NUMINAMATH_CALUDE_smallest_number_in_sample_l3992_399268


namespace NUMINAMATH_CALUDE_continuity_not_implies_differentiability_l3992_399238

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define a point in the real line
variable (x₀ : ℝ)

-- Theorem statement
theorem continuity_not_implies_differentiability :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ, ContinuousAt f x₀ ∧ ¬DifferentiableAt ℝ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_continuity_not_implies_differentiability_l3992_399238


namespace NUMINAMATH_CALUDE_conditional_probability_rhinitis_cold_l3992_399257

theorem conditional_probability_rhinitis_cold 
  (P_rhinitis : ℝ) 
  (P_rhinitis_and_cold : ℝ) 
  (h1 : P_rhinitis = 0.8) 
  (h2 : P_rhinitis_and_cold = 0.6) : 
  P_rhinitis_and_cold / P_rhinitis = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rhinitis_cold_l3992_399257


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l3992_399275

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3)
  : (total_families * total_average) / (total_families - childless_families) = 45 / 12 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l3992_399275


namespace NUMINAMATH_CALUDE_dice_line_probability_l3992_399226

-- Define the dice outcomes
def DiceOutcome : Type := Fin 6

-- Define the probability space
def Ω : Type := DiceOutcome × DiceOutcome

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event that (x, y) lies on the line 2x - y = 1
def E : Set Ω :=
  {ω : Ω | 2 * (ω.1.val + 1) - (ω.2.val + 1) = 1}

-- Theorem statement
theorem dice_line_probability :
  P E = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_dice_line_probability_l3992_399226


namespace NUMINAMATH_CALUDE_flower_arrangement_theorem_l3992_399242

/-- Represents a flower arrangement on a square -/
structure FlowerArrangement where
  corners : ℕ  -- number of flowers at each corner
  midpoints : ℕ  -- number of flowers at each midpoint

/-- The total number of flowers in the arrangement -/
def total_flowers (arrangement : FlowerArrangement) : ℕ :=
  4 * arrangement.corners + 4 * arrangement.midpoints

/-- The number of flowers seen on each side of the square -/
def flowers_per_side (arrangement : FlowerArrangement) : ℕ :=
  2 * arrangement.corners + arrangement.midpoints

theorem flower_arrangement_theorem :
  (∃ (arr : FlowerArrangement), 
    flowers_per_side arr = 9 ∧ 
    total_flowers arr = 36 ∧ 
    (∀ (other : FlowerArrangement), flowers_per_side other = 9 → total_flowers other ≤ 36)) ∧
  (∃ (arr : FlowerArrangement), 
    flowers_per_side arr = 12 ∧ 
    total_flowers arr = 24 ∧ 
    (∀ (other : FlowerArrangement), flowers_per_side other = 12 → total_flowers other ≥ 24)) :=
by sorry

end NUMINAMATH_CALUDE_flower_arrangement_theorem_l3992_399242


namespace NUMINAMATH_CALUDE_football_count_proof_l3992_399204

/-- The cost of one football -/
def football_cost : ℝ := 35

/-- The cost of one soccer ball -/
def soccer_cost : ℝ := 50

/-- The number of footballs in the first set -/
def num_footballs : ℕ := 3

/-- The total cost of the first set -/
def first_set_cost : ℝ := 155

/-- The total cost of the second set -/
def second_set_cost : ℝ := 220

theorem football_count_proof :
  (football_cost * num_footballs + soccer_cost = first_set_cost) ∧
  (2 * football_cost + 3 * soccer_cost = second_set_cost) :=
by sorry

end NUMINAMATH_CALUDE_football_count_proof_l3992_399204


namespace NUMINAMATH_CALUDE_solution_set_implies_a_l3992_399249

theorem solution_set_implies_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 6 ≤ 0 ↔ 2 ≤ x ∧ x ≤ 3) → a = -5 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_l3992_399249


namespace NUMINAMATH_CALUDE_sqrt_fourteen_times_sqrt_seven_minus_sqrt_two_l3992_399222

theorem sqrt_fourteen_times_sqrt_seven_minus_sqrt_two : 
  Real.sqrt 14 * Real.sqrt 7 - Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourteen_times_sqrt_seven_minus_sqrt_two_l3992_399222
