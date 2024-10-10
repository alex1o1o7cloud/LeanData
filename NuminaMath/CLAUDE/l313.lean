import Mathlib

namespace juanita_contest_cost_l313_31331

/-- Represents the drumming contest Juanita entered -/
structure DrummingContest where
  min_drums : ℕ  -- Minimum number of drums to hit before earning money
  earn_rate : ℚ  -- Amount earned per drum hit above min_drums
  time_limit : ℕ  -- Time limit in minutes

/-- Represents Juanita's performance in the contest -/
structure Performance where
  drums_hit : ℕ  -- Number of drums hit
  money_lost : ℚ  -- Amount of money lost (negative earnings)

def contest_entry_cost (contest : DrummingContest) (performance : Performance) : ℚ :=
  let earnings := max ((performance.drums_hit - contest.min_drums) * contest.earn_rate) 0
  earnings + performance.money_lost

theorem juanita_contest_cost :
  let contest := DrummingContest.mk 200 0.025 2
  let performance := Performance.mk 300 7.5
  contest_entry_cost contest performance = 10 := by
  sorry

end juanita_contest_cost_l313_31331


namespace min_value_of_sum_l313_31371

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) : 
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 ∧ 
  (a + 3 * b = 4 + 8 * Real.sqrt 3 ↔ a = 1 + 4 * Real.sqrt 3 ∧ b = (3 + 4 * Real.sqrt 3) / 3) := by
sorry

end min_value_of_sum_l313_31371


namespace team_photo_arrangements_l313_31334

/-- The number of students in each group (boys and girls) -/
def group_size : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := 2 * group_size

/-- The number of possible arrangements for the team photo -/
def photo_arrangements : ℕ := (Nat.factorial group_size) * (Nat.factorial group_size)

/-- Theorem stating that the number of possible arrangements is 36 -/
theorem team_photo_arrangements :
  photo_arrangements = 36 :=
by sorry

end team_photo_arrangements_l313_31334


namespace christmas_games_l313_31388

theorem christmas_games (C B : ℕ) (h1 : B = 8) (h2 : C + B + (C + B) / 2 = 30) : C = 12 := by
  sorry

end christmas_games_l313_31388


namespace smallest_n_and_b_over_a_l313_31311

theorem smallest_n_and_b_over_a : ∃ (n : ℕ+) (a b : ℝ),
  (∀ m : ℕ+, m < n → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 3*y*Complex.I)^(m:ℕ) = (x - 3*y*Complex.I)^(m:ℕ)) ∧
  a > 0 ∧ b > 0 ∧
  (a + 3*b*Complex.I)^(n:ℕ) = (a - 3*b*Complex.I)^(n:ℕ) ∧
  b/a = Real.sqrt 3 / 3 := by
sorry

end smallest_n_and_b_over_a_l313_31311


namespace specific_quadratic_equation_l313_31322

/-- A quadratic equation with given roots and leading coefficient -/
def quadratic_equation (root1 root2 : ℝ) (leading_coeff : ℝ) : ℝ → ℝ :=
  fun x => leading_coeff * (x - root1) * (x - root2)

/-- Theorem: The quadratic equation with roots -3 and 7 and leading coefficient 1 is x^2 - 4x - 21 = 0 -/
theorem specific_quadratic_equation :
  quadratic_equation (-3) 7 1 = fun x => x^2 - 4*x - 21 := by sorry

end specific_quadratic_equation_l313_31322


namespace equal_area_trapezoid_kp_l313_31398

/-- Represents a trapezoid with two bases and a point that divides it into equal areas -/
structure EqualAreaTrapezoid where
  /-- Length of the longer base KL -/
  base_kl : ℝ
  /-- Length of the shorter base MN -/
  base_mn : ℝ
  /-- Length of segment KP, where P divides the trapezoid into equal areas when connected to N -/
  kp : ℝ
  /-- Assumption that base_kl is greater than base_mn -/
  h_base : base_kl > base_mn
  /-- Assumption that all lengths are positive -/
  h_positive : base_kl > 0 ∧ base_mn > 0 ∧ kp > 0

/-- Theorem stating that for a trapezoid with given dimensions, KP = 28 when P divides the area equally -/
theorem equal_area_trapezoid_kp
  (t : EqualAreaTrapezoid)
  (h_kl : t.base_kl = 40)
  (h_mn : t.base_mn = 16) :
  t.kp = 28 := by
  sorry

#check equal_area_trapezoid_kp

end equal_area_trapezoid_kp_l313_31398


namespace hyperbola_asymptotes_l313_31393

/-- Given a hyperbola with equation y^2 - x^2/4 = 1, its asymptotes have the equation y = ± x/2 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (y^2 - x^2/4 = 1) → (∃ (k : ℝ), k = x/2 ∧ (y = k ∨ y = -k)) :=
by sorry

end hyperbola_asymptotes_l313_31393


namespace toaster_cost_is_72_l313_31399

/-- Calculates the total cost of a toaster including insurance, fees, and taxes. -/
def toaster_total_cost (msrp : ℝ) (insurance_rate : ℝ) (premium_upgrade : ℝ) 
  (recycling_fee : ℝ) (tax_rate : ℝ) : ℝ :=
  let insurance_cost := msrp * insurance_rate
  let total_insurance := insurance_cost + premium_upgrade
  let cost_before_tax := msrp + total_insurance + recycling_fee
  let tax := cost_before_tax * tax_rate
  cost_before_tax + tax

/-- Theorem stating that the total cost of the toaster is $72 given the specified conditions. -/
theorem toaster_cost_is_72 : 
  toaster_total_cost 30 0.2 7 5 0.5 = 72 := by
  sorry

end toaster_cost_is_72_l313_31399


namespace trigonometric_identity_l313_31303

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) :
  Real.cos α + 2 * Real.sin α = 1 := by
  sorry

end trigonometric_identity_l313_31303


namespace sum_base4_equals_1332_l313_31312

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c : ℕ) : ℕ := a * 4^2 + b * 4 + c

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d := n / (4^3)
  let r := n % (4^3)
  let c := r / (4^2)
  let r' := r % (4^2)
  let b := r' / 4
  let a := r' % 4
  (d, c, b, a)

theorem sum_base4_equals_1332 :
  let sum := base4ToBase10 2 1 3 + base4ToBase10 1 3 2 + base4ToBase10 3 2 1
  base10ToBase4 sum = (1, 3, 3, 2) := by sorry

end sum_base4_equals_1332_l313_31312


namespace negation_equivalence_l313_31338

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 3, x^2 - 1 ≤ 2*x) ↔
  (∀ x ∈ Set.Ioo (-1 : ℝ) 3, x^2 - 1 > 2*x) :=
by sorry

end negation_equivalence_l313_31338


namespace percent_of_12356_l313_31351

theorem percent_of_12356 (p : ℝ) : p * 12356 = 1.2356 → p * 100 = 0.01 := by
  sorry

end percent_of_12356_l313_31351


namespace trig_expression_equals_zero_l313_31348

theorem trig_expression_equals_zero :
  (Real.sin (15 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 0 := by
  sorry

end trig_expression_equals_zero_l313_31348


namespace matching_shoe_probability_l313_31390

/-- Represents a shoe pair --/
structure ShoePair :=
  (left : Nat)
  (right : Nat)

/-- The probability of selecting a matching pair of shoes --/
def probability_matching_pair (n : Nat) : Rat :=
  if n > 0 then 1 / n else 0

theorem matching_shoe_probability (cabinet : Finset ShoePair) :
  cabinet.card = 3 →
  probability_matching_pair cabinet.card = 1 / 3 := by
  sorry

#eval probability_matching_pair 3

end matching_shoe_probability_l313_31390


namespace simplify_expression_l313_31345

theorem simplify_expression (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l313_31345


namespace odd_integers_sum_21_to_65_l313_31330

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem odd_integers_sum_21_to_65 :
  arithmetic_sum 21 65 2 = 989 := by
  sorry

end odd_integers_sum_21_to_65_l313_31330


namespace round_05019_to_thousandth_l313_31358

/-- Custom rounding function that rounds to the nearest thousandth as described in the problem -/
def roundToThousandth (x : ℚ) : ℚ :=
  (⌊x * 1000⌋ : ℚ) / 1000

/-- Theorem stating that rounding 0.05019 to the nearest thousandth results in 0.050 -/
theorem round_05019_to_thousandth :
  roundToThousandth (5019 / 100000) = 50 / 1000 := by
  sorry

end round_05019_to_thousandth_l313_31358


namespace parabola_C₃_expression_l313_31344

/-- The parabola C₁ -/
def C₁ (x y : ℝ) : Prop := y = x^2 - 2*x + 3

/-- The parabola C₂, shifted 1 unit to the left from C₁ -/
def C₂ (x y : ℝ) : Prop := C₁ (x + 1) y

/-- The parabola C₃, symmetric to C₂ with respect to the y-axis -/
def C₃ (x y : ℝ) : Prop := C₂ (-x) y

/-- The theorem stating the analytical expression of C₃ -/
theorem parabola_C₃_expression : ∀ x y : ℝ, C₃ x y ↔ y = x^2 + 2 := by sorry

end parabola_C₃_expression_l313_31344


namespace fahrenheit_for_40_celsius_l313_31363

-- Define the relationship between C and F
def celsius_to_fahrenheit (C F : ℝ) : Prop :=
  C = (5/9) * (F - 32)

-- Theorem statement
theorem fahrenheit_for_40_celsius :
  ∃ F : ℝ, celsius_to_fahrenheit 40 F ∧ F = 104 :=
by
  sorry

end fahrenheit_for_40_celsius_l313_31363


namespace even_function_comparison_l313_31305

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_comparison (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m (-Real.pi) < f m 3 := by
  sorry

end even_function_comparison_l313_31305


namespace log_product_equals_two_l313_31321

theorem log_product_equals_two (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 10) = 2 →
  x = 100 := by
sorry

end log_product_equals_two_l313_31321


namespace area_is_nine_l313_31307

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Triangular region formed by two lines and x-axis -/
structure TriangularRegion where
  line1 : Line
  line2 : Line

def line1 : Line := { p1 := ⟨0, 3⟩, p2 := ⟨6, 0⟩ }
def line2 : Line := { p1 := ⟨1, 6⟩, p2 := ⟨7, 1⟩ }

def region : TriangularRegion := { line1 := line1, line2 := line2 }

/-- Calculate the area of the triangular region -/
def calculateArea (r : TriangularRegion) : ℝ :=
  sorry

theorem area_is_nine : calculateArea region = 9 := by
  sorry

end area_is_nine_l313_31307


namespace union_of_sets_l313_31341

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3}
  A ∪ B = {1, 2, 3} := by
sorry

end union_of_sets_l313_31341


namespace cube_root_of_four_fifth_power_l313_31364

theorem cube_root_of_four_fifth_power : 
  (5^7 + 5^7 + 5^7 + 5^7 : ℝ)^(1/3) = 5^(7/3) * 4^(1/3) :=
by sorry

end cube_root_of_four_fifth_power_l313_31364


namespace correct_both_problems_l313_31315

theorem correct_both_problems (total : ℕ) (sets_correct : ℕ) (functions_correct : ℕ) (both_wrong : ℕ) : 
  total = 50 ∧ sets_correct = 40 ∧ functions_correct = 31 ∧ both_wrong = 4 →
  ∃ (both_correct : ℕ), both_correct = 29 ∧ 
    total = sets_correct + functions_correct - both_correct + both_wrong :=
by
  sorry


end correct_both_problems_l313_31315


namespace correct_factorization_l313_31352

theorem correct_factorization (a : ℝ) :
  a^2 - a + (1/4 : ℝ) = (a - 1/2)^2 := by sorry

end correct_factorization_l313_31352


namespace inequality_proof_l313_31355

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_sum : a + b + c + d = 1) : 
  a^2 / (1 + a) + b^2 / (1 + b) + c^2 / (1 + c) + d^2 / (1 + d) ≥ 1/5 := by
  sorry

end inequality_proof_l313_31355


namespace leslie_garden_walkway_area_l313_31366

/-- Represents Leslie's garden layout --/
structure GardenLayout where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  row_walkway_width : Nat
  column_walkway_width : Nat

/-- Calculates the total area of walkways in the garden --/
def walkway_area (garden : GardenLayout) : Nat :=
  let total_width := garden.columns * garden.bed_width + (garden.columns + 1) * garden.column_walkway_width
  let total_height := garden.rows * garden.bed_height + (garden.rows + 1) * garden.row_walkway_width
  let total_area := total_width * total_height
  let beds_area := garden.rows * garden.columns * garden.bed_width * garden.bed_height
  total_area - beds_area

/-- Leslie's garden layout --/
def leslie_garden : GardenLayout :=
  { rows := 4
  , columns := 3
  , bed_width := 8
  , bed_height := 3
  , row_walkway_width := 1
  , column_walkway_width := 2
  }

/-- Theorem stating that the walkway area in Leslie's garden is 256 square feet --/
theorem leslie_garden_walkway_area :
  walkway_area leslie_garden = 256 := by
  sorry

end leslie_garden_walkway_area_l313_31366


namespace dream_car_mileage_difference_l313_31340

/-- Proves that the difference in miles driven between tomorrow and today is 200 miles -/
theorem dream_car_mileage_difference (consumption_rate : ℝ) (today_miles : ℝ) (total_consumption : ℝ)
  (h1 : consumption_rate = 4)
  (h2 : today_miles = 400)
  (h3 : total_consumption = 4000) :
  (total_consumption / consumption_rate) - today_miles = 200 := by
  sorry

end dream_car_mileage_difference_l313_31340


namespace strength_increase_percentage_l313_31373

/-- Calculates the percentage increase in strength due to a magical bracer --/
theorem strength_increase_percentage 
  (original_weight : ℝ) 
  (training_increase : ℝ) 
  (final_weight : ℝ) : 
  original_weight = 135 →
  training_increase = 265 →
  final_weight = 2800 →
  ((final_weight - (original_weight + training_increase)) / (original_weight + training_increase)) * 100 = 600 := by
  sorry

end strength_increase_percentage_l313_31373


namespace maggie_yellow_packs_l313_31389

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := 4

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := (total_balls - (red_packs + green_packs) * balls_per_pack) / balls_per_pack

theorem maggie_yellow_packs : yellow_packs = 8 := by
  sorry

end maggie_yellow_packs_l313_31389


namespace max_guaranteed_guesses_l313_31337

/- Define the deck of cards -/
def deck_size : Nat := 52

/- Define a function to represent the alternating arrangement of cards -/
def alternating_arrangement (i : Nat) : Bool :=
  i % 2 = 0

/- Define a function to represent the state of the deck after cutting and riffling -/
def riffled_deck (n : Nat) (i : Nat) : Bool :=
  if i < n then alternating_arrangement i else alternating_arrangement (i - n)

/- Theorem: The maximum number of guaranteed correct guesses is 26 -/
theorem max_guaranteed_guesses :
  ∀ n : Nat, n ≤ deck_size →
  ∃ strategy : Nat → Bool,
    (∀ i : Nat, i < deck_size → strategy i = riffled_deck n i) →
    (∃ correct_guesses : Nat, correct_guesses = deck_size / 2 ∧
      ∀ k : Nat, k < correct_guesses → strategy k = riffled_deck n k) :=
by sorry

end max_guaranteed_guesses_l313_31337


namespace urn_probability_l313_31384

/-- Represents the contents of the urn -/
structure UrnContents where
  red : ℕ
  blue : ℕ

/-- Represents a single operation of drawing and adding balls -/
inductive Operation
  | DrawRed
  | DrawBlue

/-- The initial state of the urn -/
def initial_urn : UrnContents := ⟨2, 1⟩

/-- Perform a single operation on the urn -/
def perform_operation (urn : UrnContents) (op : Operation) : UrnContents :=
  match op with
  | Operation.DrawRed => ⟨urn.red + 2, urn.blue⟩
  | Operation.DrawBlue => ⟨urn.red, urn.blue + 2⟩

/-- Perform a sequence of operations on the urn -/
def perform_operations (urn : UrnContents) (ops : List Operation) : UrnContents :=
  ops.foldl perform_operation urn

/-- Calculate the probability of a specific sequence of operations -/
def sequence_probability (ops : List Operation) : ℚ :=
  sorry

/-- Calculate the total probability of all valid sequences -/
def total_probability (valid_sequences : List (List Operation)) : ℚ :=
  sorry

theorem urn_probability : 
  ∃ (valid_sequences : List (List Operation)),
    (∀ seq ∈ valid_sequences, seq.length = 5) ∧
    (∀ seq ∈ valid_sequences, 
      let final_urn := perform_operations initial_urn seq
      final_urn.red + final_urn.blue = 12 ∧
      final_urn.red = 7 ∧ final_urn.blue = 5) ∧
    total_probability valid_sequences = 25 / 224 :=
  sorry

end urn_probability_l313_31384


namespace f_has_unique_zero_in_interval_l313_31323

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem f_has_unique_zero_in_interval :
  ∃! x, x ∈ (Set.Ioo 0 (1/2)) ∧ f x = 0 :=
sorry

end f_has_unique_zero_in_interval_l313_31323


namespace vanessa_chocolate_sales_l313_31309

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that Vanessa made $16 from selling chocolate bars -/
theorem vanessa_chocolate_sales :
  money_made 11 4 7 = 16 := by
  sorry

end vanessa_chocolate_sales_l313_31309


namespace volume_of_solid_T_l313_31308

/-- The solid T is defined as the set of all points (x, y, z) in ℝ³ that satisfy
    the given inequalities. -/
def solid_T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p
    (|x| + |y| ≤ 1.5) ∧ (|x| + |z| ≤ 1) ∧ (|y| + |z| ≤ 1)}

/-- The volume of a set in ℝ³. -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of solid T is 2/3. -/
theorem volume_of_solid_T : volume solid_T = 2/3 := by sorry

end volume_of_solid_T_l313_31308


namespace product_of_max_min_sum_l313_31372

theorem product_of_max_min_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * 2^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0 →
  ∃ (min_sum max_sum : ℝ),
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * 2^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 →
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 :=
by sorry

end product_of_max_min_sum_l313_31372


namespace quadratic_equation_root_l313_31361

/-- Given a quadratic equation (m+2)x^2 - x + m^2 - 4 = 0 where one root is 0,
    prove that the other root is 1/4 -/
theorem quadratic_equation_root (m : ℝ) :
  (∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x = 0) →
  (∃ y : ℝ, (m + 2) * y^2 - y + m^2 - 4 = 0 ∧ y = 1/4) :=
by sorry

end quadratic_equation_root_l313_31361


namespace dinosaur_count_correct_l313_31350

/-- Represents the number of dinosaurs in the flock -/
def num_dinosaurs : ℕ := 5

/-- Represents the number of legs each dinosaur has -/
def legs_per_dinosaur : ℕ := 3

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 20

/-- Proves that the number of dinosaurs in the flock is correct -/
theorem dinosaur_count_correct :
  num_dinosaurs * (legs_per_dinosaur + 1) = total_heads_and_legs :=
by sorry

end dinosaur_count_correct_l313_31350


namespace scientist_contemporary_probability_scientist_contemporary_probability_value_l313_31317

/-- The probability that two scientists were contemporaries for any length of time -/
theorem scientist_contemporary_probability : ℝ :=
  let years_range : ℕ := 300
  let lifespan : ℕ := 80
  let total_possibility_area : ℕ := years_range * years_range
  let overlap_area : ℕ := (years_range - lifespan) * (years_range - lifespan) - 2 * (lifespan * lifespan / 2)
  (overlap_area : ℝ) / total_possibility_area

/-- The probability is equal to 7/15 -/
theorem scientist_contemporary_probability_value : scientist_contemporary_probability = 7 / 15 :=
sorry

end scientist_contemporary_probability_scientist_contemporary_probability_value_l313_31317


namespace floor_sqrt_50_squared_l313_31343

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end floor_sqrt_50_squared_l313_31343


namespace boys_to_girls_ratio_l313_31302

theorem boys_to_girls_ratio (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 48) (h2 : boys = 42) : 
  (boys : ℚ) / (total_students - boys : ℚ) = 7 / 1 := by
  sorry

end boys_to_girls_ratio_l313_31302


namespace sum_of_absolute_b_values_l313_31360

-- Define the polynomials p and q
def p (a b x : ℝ) : ℝ := x^3 + a*x + b
def q (a b x : ℝ) : ℝ := x^3 + a*x + b + 240

-- Define the theorem
theorem sum_of_absolute_b_values (a b r s : ℝ) : 
  (p a b r = 0) → 
  (p a b s = 0) → 
  (q a b (r + 4) = 0) → 
  (q a b (s - 3) = 0) → 
  (∃ b₁ b₂ : ℝ, (b = b₁ ∨ b = b₂) ∧ (|b₁| + |b₂| = 62)) := by
sorry


end sum_of_absolute_b_values_l313_31360


namespace smallest_sum_B_plus_b_l313_31353

theorem smallest_sum_B_plus_b :
  ∀ (B b : ℕ),
  0 ≤ B ∧ B < 5 →
  b > 6 →
  31 * B = 4 * b + 4 →
  ∀ (B' b' : ℕ),
  0 ≤ B' ∧ B' < 5 →
  b' > 6 →
  31 * B' = 4 * b' + 4 →
  B + b ≤ B' + b' :=
sorry

end smallest_sum_B_plus_b_l313_31353


namespace max_second_term_arithmetic_sequence_l313_31332

theorem max_second_term_arithmetic_sequence : ∀ (a d : ℕ),
  a > 0 ∧ d > 0 ∧ 
  a + (a + d) + (a + 2*d) + (a + 3*d) = 52 →
  a + d ≤ 10 :=
by sorry

end max_second_term_arithmetic_sequence_l313_31332


namespace fence_perimeter_is_236_l313_31378

/-- Represents the configuration of a rectangular fence --/
structure FenceConfig where
  total_posts : ℕ
  post_width : ℚ
  gap_width : ℕ
  long_short_ratio : ℕ

/-- Calculates the perimeter of the fence given its configuration --/
def calculate_perimeter (config : FenceConfig) : ℚ :=
  let short_side_posts := config.total_posts / (config.long_short_ratio + 1)
  let long_side_posts := short_side_posts * config.long_short_ratio
  let short_side_length := short_side_posts * config.post_width + (short_side_posts - 1) * config.gap_width
  let long_side_length := long_side_posts * config.post_width + (long_side_posts - 1) * config.gap_width
  2 * (short_side_length + long_side_length)

/-- The main theorem stating that the fence configuration results in a perimeter of 236 feet --/
theorem fence_perimeter_is_236 :
  let config : FenceConfig := {
    total_posts := 36,
    post_width := 1/2,  -- 6 inches = 1/2 foot
    gap_width := 6,
    long_short_ratio := 3
  }
  calculate_perimeter config = 236 := by sorry


end fence_perimeter_is_236_l313_31378


namespace power_mod_prime_remainder_5_1000_mod_29_l313_31374

theorem power_mod_prime (p : Nat) (a : Nat) (m : Nat) (h : Prime p) :
  a ^ m % p = (a ^ (m % (p - 1))) % p :=
sorry

theorem remainder_5_1000_mod_29 : 5^1000 % 29 = 21 := by
  have h1 : Prime 29 := by sorry
  have h2 : 5^1000 % 29 = (5^(1000 % 28)) % 29 := by
    apply power_mod_prime 29 5 1000 h1
  have h3 : 1000 % 28 = 20 := by sorry
  have h4 : (5^20) % 29 = 21 := by sorry
  rw [h2, h3, h4]

end power_mod_prime_remainder_5_1000_mod_29_l313_31374


namespace oil_cylinder_capacity_l313_31362

theorem oil_cylinder_capacity (capacity : ℚ) 
  (h1 : (3 : ℚ) / 4 * capacity = 27.5)
  (h2 : (9 : ℚ) / 10 * capacity = 35) : 
  capacity = 110 / 3 := by sorry

end oil_cylinder_capacity_l313_31362


namespace locus_of_point_on_moving_segment_l313_31346

/-- 
Given two perpendicular lines with moving points M and N, where the distance MN 
remains constant, and P is an arbitrary point on segment MN, prove that the locus 
of points P(x,y) forms an ellipse described by the equation x²/a² + y²/b² = 1, 
where a and b are constants related to the distance MN.
-/
theorem locus_of_point_on_moving_segment (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (M N P : ℝ × ℝ) (dist_MN : ℝ),
    (∀ t : ℝ, ∃ (Mt Nt : ℝ × ℝ), 
      (Mt.1 = 0 ∧ Nt.2 = 0) ∧  -- M and N move on perpendicular lines
      (Mt.2 - Nt.2)^2 + (Mt.1 - Nt.1)^2 = dist_MN^2 ∧  -- constant distance MN
      (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 
        P = (s * Mt.1 + (1 - s) * Nt.1, s * Mt.2 + (1 - s) * Nt.2))) →  -- P on segment MN
    P.1^2 / a^2 + P.2^2 / b^2 = 1  -- locus is an ellipse
  := by sorry

end locus_of_point_on_moving_segment_l313_31346


namespace walkers_meet_at_corner_d_l313_31379

/-- Represents the corners of the rectangular area -/
inductive Corner
| A
| B
| C
| D

/-- Represents the rectangular area -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a person walking along the perimeter -/
structure Walker where
  speed : ℚ
  startCorner : Corner
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The meeting point of two walkers -/
def meetingPoint (rect : Rectangle) (walker1 walker2 : Walker) : Corner :=
  sorry

/-- The theorem to be proved -/
theorem walkers_meet_at_corner_d 
  (rect : Rectangle)
  (jane hector : Walker)
  (h_rect_dims : rect.length = 10 ∧ rect.width = 4)
  (h_start : jane.startCorner = Corner.A ∧ hector.startCorner = Corner.A)
  (h_directions : jane.direction = false ∧ hector.direction = true)
  (h_speeds : jane.speed = 2 * hector.speed) :
  meetingPoint rect jane hector = Corner.D :=
sorry

end walkers_meet_at_corner_d_l313_31379


namespace quadratic_equation_solution_l313_31339

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 2*x = 0 ↔ x = 0 ∨ x = -2 := by sorry

end quadratic_equation_solution_l313_31339


namespace birds_on_trees_l313_31397

theorem birds_on_trees (n : ℕ) (h : n = 44) : 
  let initial_sum := n * (n + 1) / 2
  ∀ (current_sum : ℕ), current_sum % 4 ≠ 0 →
    ∃ (next_sum : ℕ), (next_sum = current_sum ∨ next_sum = current_sum + n - 1 ∨ next_sum = current_sum - (n - 1)) ∧
      next_sum % 4 ≠ 0 :=
by sorry

#check birds_on_trees

end birds_on_trees_l313_31397


namespace right_triangle_hypotenuse_l313_31377

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 7 → b = 24 → c^2 = a^2 + b^2 → c = 25 :=
by sorry

end right_triangle_hypotenuse_l313_31377


namespace completing_square_equivalence_l313_31327

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) := by
  sorry

end completing_square_equivalence_l313_31327


namespace quadratic_solution_l313_31375

theorem quadratic_solution : ∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = -1 ∧ ∀ x : ℝ, x^2 - 5*x - 6 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end quadratic_solution_l313_31375


namespace eric_required_bike_speed_l313_31310

/-- Represents the triathlon components --/
structure Triathlon :=
  (swim_distance : ℚ)
  (swim_speed : ℚ)
  (run_distance : ℚ)
  (run_speed : ℚ)
  (bike_distance : ℚ)
  (total_time : ℚ)

/-- Calculates the required bike speed for a given triathlon --/
def required_bike_speed (t : Triathlon) : ℚ :=
  let swim_time := t.swim_distance / t.swim_speed
  let run_time := t.run_distance / t.run_speed
  let bike_time := t.total_time - (swim_time + run_time)
  t.bike_distance / bike_time

/-- The triathlon problem --/
def eric_triathlon : Triathlon :=
  { swim_distance := 1/4
  , swim_speed := 2
  , run_distance := 3
  , run_speed := 6
  , bike_distance := 15
  , total_time := 2 }

/-- Theorem stating that the required bike speed for Eric's triathlon is 120/11 --/
theorem eric_required_bike_speed :
  required_bike_speed eric_triathlon = 120/11 := by sorry


end eric_required_bike_speed_l313_31310


namespace banana_arrangements_l313_31391

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end banana_arrangements_l313_31391


namespace pieces_after_n_divisions_no_2009_pieces_l313_31300

/-- Represents the number of pieces after n divisions -/
def num_pieces (n : ℕ) : ℕ := 3 * n + 1

/-- Theorem stating the number of pieces after n divisions -/
theorem pieces_after_n_divisions (n : ℕ) :
  num_pieces n = 3 * n + 1 := by sorry

/-- Theorem stating that it's impossible to have 2009 pieces -/
theorem no_2009_pieces :
  ¬ ∃ (n : ℕ), num_pieces n = 2009 := by sorry

end pieces_after_n_divisions_no_2009_pieces_l313_31300


namespace combined_machine_time_order_completion_time_l313_31357

theorem combined_machine_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  1 / (1 / t1 + 1 / t2) = (t1 * t2) / (t1 + t2) := by sorry

theorem order_completion_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) :
  t1 = 20 → t2 = 30 → 1 / (1 / t1 + 1 / t2) = 12 := by sorry

end combined_machine_time_order_completion_time_l313_31357


namespace probability_divisible_by_15_l313_31354

def digits : Finset ℕ := {1, 2, 3, 5, 5, 8}

def is_valid_arrangement (n : ℕ) : Prop :=
  n ≥ 100000 ∧ n < 1000000 ∧ ∀ d, d ∈ digits.toList.map Nat.digitChar → d ∈ n.repr.data

def is_divisible_by_15 (n : ℕ) : Prop := n % 15 = 0

def total_arrangements : ℕ := 6 * 5 * 4 * 3 * 2 * 1

def favorable_arrangements : ℕ := 2 * (5 * 4 * 3 * 2 * 1)

theorem probability_divisible_by_15 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 :=
sorry

end probability_divisible_by_15_l313_31354


namespace ones_digit_73_pow_351_l313_31369

theorem ones_digit_73_pow_351 : (73^351) % 10 = 7 := by
  sorry

end ones_digit_73_pow_351_l313_31369


namespace complement_intersection_equals_four_l313_31342

-- Define the universe
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets M and N
def M : Set Nat := {1, 3, 5}
def N : Set Nat := {3, 4, 5}

-- State the theorem
theorem complement_intersection_equals_four :
  (U \ M) ∩ N = {4} := by sorry

end complement_intersection_equals_four_l313_31342


namespace rationalize_denominator_sqrt5_l313_31306

theorem rationalize_denominator_sqrt5 : 
  ∃ (A B C : ℤ), 
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧ 
    A * B * C = 275 := by
  sorry

end rationalize_denominator_sqrt5_l313_31306


namespace locus_is_circumcircle_l313_31380

/-- Triangle represented by its vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Distance from a point to a line segment -/
def distToSide (P : Point) (A B : Point) : ℝ := sorry

/-- Distance between two points -/
def dist (P Q : Point) : ℝ := sorry

/-- Circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set Point := sorry

/-- A point lies on the circumcircle of a triangle -/
def onCircumcircle (P : Point) (t : Triangle) : Prop :=
  P ∈ circumcircle t

theorem locus_is_circumcircle (t : Triangle) (P : Point) :
  (distToSide P t.A t.B * dist P t.C = distToSide P t.A t.C * dist P t.B) →
  onCircumcircle P t := by
  sorry

end locus_is_circumcircle_l313_31380


namespace cos_identity_l313_31326

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 - α) = 3 / 5) :
  Real.cos (5 * π / 6 + α) = -(3 / 5) := by
  sorry

end cos_identity_l313_31326


namespace continuous_and_strictly_monotone_function_l313_31365

-- Define a function type from reals to reals
def RealFunction := ℝ → ℝ

-- Define the property of having limits at any point
def has_limits_at_any_point (f : RealFunction) : Prop :=
  ∀ a : ℝ, ∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - a| < δ → |f x - L| < ε

-- Define the property of having no local extrema
def has_no_local_extrema (f : RealFunction) : Prop :=
  ∀ a : ℝ, ∀ ε > 0, ∃ x y : ℝ, |x - a| < ε ∧ |y - a| < ε ∧ f x < f a ∧ f a < f y

-- State the theorem
theorem continuous_and_strictly_monotone_function 
  (f : RealFunction) 
  (h1 : has_limits_at_any_point f) 
  (h2 : has_no_local_extrema f) : 
  Continuous f ∧ StrictMono f :=
by sorry

end continuous_and_strictly_monotone_function_l313_31365


namespace find_x_l313_31336

theorem find_x : 
  ∀ x : ℝ, 
  (x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2400.0000000000005 → 
  x = 10.8 := by
sorry

end find_x_l313_31336


namespace a_51_value_l313_31376

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem a_51_value (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end a_51_value_l313_31376


namespace pure_imaginary_condition_l313_31349

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (Complex.ofReal (a - 1) * Complex.ofReal (a + 1) + Complex.I) = 
    Complex.ofReal (a^2 - 1) + Complex.I * Complex.ofReal (a - 1) →
  (Complex.ofReal (a - 1) * Complex.ofReal (a + 1) + Complex.I).re = 0 →
  a = 1 := by
sorry

end pure_imaginary_condition_l313_31349


namespace new_student_weights_l313_31367

/-- Proves that given the class size changes and average weights, the weights of the four new students are as calculated. -/
theorem new_student_weights
  (original_size : ℕ)
  (original_avg : ℝ)
  (avg_after_first : ℝ)
  (avg_after_second : ℝ)
  (avg_after_third : ℝ)
  (final_avg : ℝ)
  (h_original_size : original_size = 29)
  (h_original_avg : original_avg = 28)
  (h_avg_after_first : avg_after_first = 27.2)
  (h_avg_after_second : avg_after_second = 27.8)
  (h_avg_after_third : avg_after_third = 27.6)
  (h_final_avg : final_avg = 28) :
  ∃ (w1 w2 w3 w4 : ℝ),
    w1 = 4 ∧
    w2 = 45.8 ∧
    w3 = 21.4 ∧
    w4 = 40.8 ∧
    (original_size : ℝ) * original_avg + w1 = (original_size + 1 : ℝ) * avg_after_first ∧
    (original_size + 1 : ℝ) * avg_after_first + w2 = (original_size + 2 : ℝ) * avg_after_second ∧
    (original_size + 2 : ℝ) * avg_after_second + w3 = (original_size + 3 : ℝ) * avg_after_third ∧
    (original_size + 3 : ℝ) * avg_after_third + w4 = (original_size + 4 : ℝ) * final_avg :=
by
  sorry

end new_student_weights_l313_31367


namespace remainder_preserving_operation_l313_31324

theorem remainder_preserving_operation (N : ℤ) (f : ℤ → ℤ) :
  N % 6 = 3 → f N % 6 = 3 →
  ∃ k : ℤ, f N = N + 6 * k :=
sorry

end remainder_preserving_operation_l313_31324


namespace complex_on_imaginary_axis_l313_31392

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) * (1 + a * Complex.I)
  (z.re = 0) → a = 1 := by
  sorry

end complex_on_imaginary_axis_l313_31392


namespace figure2_segment_length_l313_31328

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Calculates the total length of visible segments after cutting a square from a rectangle -/
def visibleSegmentsLength (rect : Rectangle) (sq : Square) : ℝ :=
  rect.length + (rect.width - sq.side) + (rect.length - sq.side) + sq.side

/-- Theorem stating that the total length of visible segments in Figure 2 is 23 units -/
theorem figure2_segment_length :
  let rect : Rectangle := { length := 10, width := 6 }
  let sq : Square := { side := 3 }
  visibleSegmentsLength rect sq = 23 := by
  sorry

end figure2_segment_length_l313_31328


namespace range_of_x_when_a_is_neg_one_range_of_a_for_p_sufficient_not_necessary_for_q_l313_31320

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x - 72 ≤ 0 ∧ x^2 + x - 6 > 0

-- Part 1: Range of x when a = -1
theorem range_of_x_when_a_is_neg_one :
  ∀ x : ℝ, (p x (-1) ∨ q x) ↔ x ∈ Set.Ioc (-6) (-3) ∪ Set.Icc 1 12 :=
sorry

-- Part 2: Range of a when p is sufficient but not necessary for q
theorem range_of_a_for_p_sufficient_not_necessary_for_q :
  {a : ℝ | ∀ x : ℝ, p x a → q x} ∩ {a : ℝ | ∃ x : ℝ, q x ∧ ¬p x a} = Set.Icc (-4) (-2) :=
sorry

end range_of_x_when_a_is_neg_one_range_of_a_for_p_sufficient_not_necessary_for_q_l313_31320


namespace f_one_zero_iff_l313_31329

/-- A function f(x) = ax^2 - x - 1 where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - 1

/-- The property that f has exactly one zero -/
def has_exactly_one_zero (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that f has exactly one zero iff a = 0 or a = -1/4 -/
theorem f_one_zero_iff (a : ℝ) :
  has_exactly_one_zero a ↔ a = 0 ∨ a = -1/4 := by sorry

end f_one_zero_iff_l313_31329


namespace log_equation_sum_l313_31387

theorem log_equation_sum (a b : ℤ) (h : a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3) : 
  a + 2 * b = 21 := by
  sorry

end log_equation_sum_l313_31387


namespace exists_non_increasing_function_with_condition_increasing_on_subset_not_implies_entire_interval_inverse_function_decreasing_intervals_monotonic_function_extrema_on_closed_interval_l313_31333

-- 1
theorem exists_non_increasing_function_with_condition :
  ∃ f : ℝ → ℝ, f (-1) < f 3 ∧ ¬(∀ x y : ℝ, x < y → f x < f y) := by sorry

-- 2
theorem increasing_on_subset_not_implies_entire_interval
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) :
  ¬(∀ a b : ℝ, a < b → (∀ x y : ℝ, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x < f y) →
    Set.Icc a b = Set.Ici 1) := by sorry

-- 3
theorem inverse_function_decreasing_intervals :
  ¬(∀ x y : ℝ, (x < y ∧ x < 0 ∧ y < 0) ∨ (x < y ∧ x > 0 ∧ y > 0) → 1/x > 1/y) := by sorry

-- 4
theorem monotonic_function_extrema_on_closed_interval
  {a b : ℝ} (f : ℝ → ℝ) (h : Monotone f) :
  ∃ x y : ℝ, x ∈ Set.Icc a b ∧ y ∈ Set.Icc a b ∧
    (∀ z : ℝ, z ∈ Set.Icc a b → f x ≤ f z) ∧
    (∀ z : ℝ, z ∈ Set.Icc a b → f z ≤ f y) ∧
    (x = a ∨ x = b) ∧ (y = a ∨ y = b) := by sorry

end exists_non_increasing_function_with_condition_increasing_on_subset_not_implies_entire_interval_inverse_function_decreasing_intervals_monotonic_function_extrema_on_closed_interval_l313_31333


namespace recreation_spending_comparison_l313_31325

theorem recreation_spending_comparison 
  (last_week_wages : ℝ) 
  (last_week_recreation_percent : ℝ) 
  (this_week_wage_reduction : ℝ) 
  (this_week_recreation_percent : ℝ) 
  (h1 : last_week_recreation_percent = 0.1)
  (h2 : this_week_wage_reduction = 0.1)
  (h3 : this_week_recreation_percent = 0.4) :
  (this_week_recreation_percent * (1 - this_week_wage_reduction) * last_week_wages) / 
  (last_week_recreation_percent * last_week_wages) * 100 = 360 := by
  sorry

end recreation_spending_comparison_l313_31325


namespace sin_cos_identity_l313_31396

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l313_31396


namespace solve_inequality_find_range_of_a_l313_31395

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Theorem for part (1)
theorem solve_inequality (x : ℝ) :
  let a : ℝ := 3
  f x a > g a + 2 ↔ x < -4 ∨ x > 2 := by sorry

-- Theorem for part (2)
theorem find_range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) ↔ a ≥ 3 := by sorry

end solve_inequality_find_range_of_a_l313_31395


namespace reciprocal_sum_l313_31383

theorem reciprocal_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 6 := by
sorry

end reciprocal_sum_l313_31383


namespace prob_no_consecutive_heads_is_half_l313_31386

/-- The probability of heads not appearing consecutively when tossing a fair coin four times -/
def prob_no_consecutive_heads : ℚ := 1/2

/-- A fair coin is tossed four times -/
def num_tosses : ℕ := 4

/-- The total number of possible outcomes when tossing a fair coin four times -/
def total_outcomes : ℕ := 2^num_tosses

/-- The number of outcomes where heads do not appear consecutively -/
def favorable_outcomes : ℕ := 8

theorem prob_no_consecutive_heads_is_half :
  prob_no_consecutive_heads = favorable_outcomes / total_outcomes :=
by sorry

end prob_no_consecutive_heads_is_half_l313_31386


namespace derek_added_water_l313_31347

theorem derek_added_water (initial_amount final_amount : ℝ) (h1 : initial_amount = 3) (h2 : final_amount = 9.8) :
  final_amount - initial_amount = 6.8 := by
  sorry

end derek_added_water_l313_31347


namespace friend_lunch_cost_l313_31370

theorem friend_lunch_cost (total : ℕ) (difference : ℕ) (friend_cost : ℕ) : 
  total = 15 → difference = 5 → 
  (∃ (your_cost : ℕ), your_cost + friend_cost = total ∧ friend_cost = your_cost + difference) →
  friend_cost = 10 := by sorry

end friend_lunch_cost_l313_31370


namespace triangle_similarity_theorem_l313_31359

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the line segment MN
structure LineSegment :=
  (M N : ℝ × ℝ)

-- Define the parallel property
def isParallel (l1 l2 : LineSegment) : Prop := sorry

-- Define the length of a line segment
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_theorem (XYZ : Triangle) (MN : LineSegment) :
  isParallel MN (LineSegment.mk XYZ.X XYZ.Y) →
  length XYZ.X MN.M = 5 →
  length MN.M XYZ.Y = 8 →
  length MN.N XYZ.Z = 9 →
  length XYZ.Y XYZ.Z = 23.4 := by
  sorry

end triangle_similarity_theorem_l313_31359


namespace alligator_population_after_year_l313_31318

/-- The number of alligators after a given number of 6-month periods -/
def alligator_population (initial_population : ℕ) (periods : ℕ) : ℕ :=
  initial_population * 2^periods

/-- Theorem stating that given 4 initial alligators and population doubling every 6 months, 
    there will be 16 alligators after 1 year -/
theorem alligator_population_after_year (initial_population : ℕ) 
  (h1 : initial_population = 4) : alligator_population initial_population 2 = 16 := by
  sorry

#check alligator_population_after_year

end alligator_population_after_year_l313_31318


namespace polynomial_root_problem_l313_31335

theorem polynomial_root_problem (a b c d : ℝ) 
  (h1 : ∃ x1 x2 x3 x4 : ℂ, x1^4 + a*x1^3 + b*x1^2 + c*x1 + d = 0 ∧ 
                           x2^4 + a*x2^3 + b*x2^2 + c*x2 + d = 0 ∧ 
                           x3^4 + a*x3^3 + b*x3^2 + c*x3 + d = 0 ∧ 
                           x4^4 + a*x4^3 + b*x4^2 + c*x4 + d = 0)
  (h2 : ∀ x : ℂ, x^4 + a*x^3 + b*x^2 + c*x + d = 0 → x.im ≠ 0)
  (h3 : ∃ x1 x2 : ℂ, (x1^4 + a*x1^3 + b*x1^2 + c*x1 + d = 0) ∧ 
                     (x2^4 + a*x2^3 + b*x2^2 + c*x2 + d = 0) ∧ 
                     (x1 * x2 = 13 + I))
  (h4 : ∃ x3 x4 : ℂ, (x3^4 + a*x3^3 + b*x3^2 + c*x3 + d = 0) ∧ 
                     (x4^4 + a*x4^3 + b*x4^2 + c*x4 + d = 0) ∧ 
                     (x3 + x4 = 3 + 4*I)) :
  b = 51 := by sorry

end polynomial_root_problem_l313_31335


namespace farm_plot_length_l313_31385

/-- Proves that a rectangular plot with given width and area has a specific length -/
theorem farm_plot_length (width : ℝ) (area_acres : ℝ) (acre_sq_ft : ℝ) :
  width = 1210 →
  area_acres = 10 →
  acre_sq_ft = 43560 →
  (area_acres * acre_sq_ft) / width = 360 := by
  sorry

end farm_plot_length_l313_31385


namespace other_root_of_quadratic_l313_31314

theorem other_root_of_quadratic (c : ℝ) : 
  (3 : ℝ) ∈ {x : ℝ | x^2 - 5*x + c = 0} → 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5*x + c = 0 ∧ x = 2 :=
by sorry

end other_root_of_quadratic_l313_31314


namespace complex_fraction_simplification_l313_31316

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((4 + 7 * i) / (4 - 7 * i) + (4 - 7 * i) / (4 + 7 * i)) = 2 := by
  sorry

end complex_fraction_simplification_l313_31316


namespace turner_ticket_count_l313_31381

/-- The number of times Turner wants to ride the rollercoaster -/
def rollercoaster_rides : ℕ := 3

/-- The number of times Turner wants to ride the Catapult -/
def catapult_rides : ℕ := 2

/-- The number of times Turner wants to ride the Ferris wheel -/
def ferris_wheel_rides : ℕ := 1

/-- The number of tickets required for one rollercoaster ride -/
def rollercoaster_cost : ℕ := 4

/-- The number of tickets required for one Catapult ride -/
def catapult_cost : ℕ := 4

/-- The number of tickets required for one Ferris wheel ride -/
def ferris_wheel_cost : ℕ := 1

/-- The total number of tickets Turner needs -/
def total_tickets : ℕ := 
  rollercoaster_rides * rollercoaster_cost + 
  catapult_rides * catapult_cost + 
  ferris_wheel_rides * ferris_wheel_cost

theorem turner_ticket_count : total_tickets = 21 := by
  sorry

end turner_ticket_count_l313_31381


namespace max_additional_plates_l313_31304

/-- Represents the number of elements in each set of characters for license plates -/
def initial_sets : Fin 3 → Nat
  | 0 => 4  -- {B, G, J, S}
  | 1 => 2  -- {E, U}
  | 2 => 3  -- {K, V, X}
  | _ => 0

/-- Calculates the total number of license plate combinations -/
def total_combinations (sets : Fin 3 → Nat) : Nat :=
  (sets 0) * (sets 1) * (sets 2)

/-- Represents the addition of two new letters to the sets -/
structure NewLetterAddition where
  set1 : Nat  -- Number of letters added to set 1
  set2 : Nat  -- Number of letters added to set 2
  set3 : Nat  -- Number of letters added to set 3

/-- The theorem to be proved -/
theorem max_additional_plates :
  ∃ (addition : NewLetterAddition),
    addition.set1 + addition.set2 + addition.set3 = 2 ∧
    ∀ (other : NewLetterAddition),
      other.set1 + other.set2 + other.set3 = 2 →
      total_combinations (λ i => initial_sets i + other.set1) -
      total_combinations initial_sets ≤
      total_combinations (λ i => initial_sets i + addition.set1) -
      total_combinations initial_sets ∧
      total_combinations (λ i => initial_sets i + addition.set1) -
      total_combinations initial_sets = 24 :=
sorry

end max_additional_plates_l313_31304


namespace third_candidate_votes_l313_31313

theorem third_candidate_votes (total_votes : ℕ) 
  (h1 : total_votes = 52500)
  (h2 : ∃ (c1 c2 c3 : ℕ), c1 + c2 + c3 = total_votes ∧ c1 = 2500 ∧ c2 = 15000)
  (h3 : ∃ (winner : ℕ), winner = (2 : ℚ) / 3 * total_votes) :
  ∃ (third : ℕ), third = 35000 := by
sorry

end third_candidate_votes_l313_31313


namespace average_boxes_theorem_l313_31382

def boxes_day1 : ℕ := 318
def boxes_day2 : ℕ := 312
def boxes_day3_part1 : ℕ := 180
def boxes_day3_part2 : ℕ := 162
def total_days : ℕ := 3

def average_boxes_per_day : ℚ :=
  (boxes_day1 + boxes_day2 + boxes_day3_part1 + boxes_day3_part2) / total_days

theorem average_boxes_theorem : average_boxes_per_day = 324 := by
  sorry

end average_boxes_theorem_l313_31382


namespace triangle_angle_b_value_l313_31319

/-- Given a triangle ABC with side lengths a and b, and angle A, proves that angle B has a specific value. -/
theorem triangle_angle_b_value 
  (a b : ℝ) 
  (A B : ℝ) 
  (h1 : a = 2 * Real.sqrt 3)
  (h2 : b = Real.sqrt 6)
  (h3 : A = π/4)  -- 45° in radians
  (h4 : 0 < A ∧ A < π)  -- A is a valid angle
  (h5 : 0 < B ∧ B < π)  -- B is a valid angle
  : B = π/6  -- 30° in radians
:= by sorry

end triangle_angle_b_value_l313_31319


namespace log_function_value_l313_31356

-- Define the logarithmic function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_function_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a (1/8) = 3) → (f a (1/4) = 2) :=
by sorry

end log_function_value_l313_31356


namespace max_temperature_range_l313_31394

theorem max_temperature_range (temps : Finset ℝ) (avg : ℝ) (min_temp : ℝ) :
  temps.card = 5 →
  Finset.sum temps id / temps.card = avg →
  avg = 60 →
  min_temp = 40 →
  min_temp ∈ temps →
  ∀ t ∈ temps, t ≥ min_temp →
  ∃ max_temp ∈ temps, max_temp - min_temp ≤ 100 ∧
    ∀ t ∈ temps, t - min_temp ≤ max_temp - min_temp :=
by sorry

end max_temperature_range_l313_31394


namespace airplane_seats_l313_31301

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) : 
  total_seats = 567 → 
  first_class + 3 * first_class + (7 * first_class + 5) = total_seats →
  7 * first_class + 5 = 362 := by
sorry

end airplane_seats_l313_31301


namespace odd_rolls_probability_l313_31368

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has a success probability of p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of rolls of the die -/
def num_rolls : ℕ := 7

/-- The number of odd outcomes we're interested in -/
def num_odd : ℕ := 5

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

theorem odd_rolls_probability :
  binomial_probability num_rolls num_odd prob_odd = 21/128 := by
  sorry

end odd_rolls_probability_l313_31368
