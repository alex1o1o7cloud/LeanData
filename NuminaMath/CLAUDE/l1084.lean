import Mathlib

namespace NUMINAMATH_CALUDE_crayon_count_l1084_108433

theorem crayon_count (initial_crayons added_crayons : ℕ) 
  (h1 : initial_crayons = 9)
  (h2 : added_crayons = 3) :
  initial_crayons + added_crayons = 12 := by
sorry

end NUMINAMATH_CALUDE_crayon_count_l1084_108433


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1084_108448

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1084_108448


namespace NUMINAMATH_CALUDE_quadratic_polynomial_unique_l1084_108419

theorem quadratic_polynomial_unique (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →
  q (-4) = 17 →
  q 1 = 2 →
  q 3 = 10 →
  ∀ x, q x = x^2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_unique_l1084_108419


namespace NUMINAMATH_CALUDE_ratio_equality_l1084_108497

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) :
  (a / 3) / (b / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1084_108497


namespace NUMINAMATH_CALUDE_multiply_24_99_l1084_108451

theorem multiply_24_99 : 24 * 99 = 2376 := by
  sorry

end NUMINAMATH_CALUDE_multiply_24_99_l1084_108451


namespace NUMINAMATH_CALUDE_max_value_of_f_l1084_108416

-- Define the function
def f (x : ℝ) : ℝ := x * abs x - 2 * x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 16 ∧ ∀ x : ℝ, |x + 1| ≤ 6 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1084_108416


namespace NUMINAMATH_CALUDE_kids_in_restaurant_group_l1084_108496

/-- Represents the number of kids in a restaurant group given certain conditions. -/
def number_of_kids (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) : ℕ :=
  total_people - (total_cost / adult_meal_cost)

/-- Theorem stating that given the problem conditions, the number of kids is 9. -/
theorem kids_in_restaurant_group :
  let total_people : ℕ := 13
  let adult_meal_cost : ℕ := 7
  let total_cost : ℕ := 28
  number_of_kids total_people adult_meal_cost total_cost = 9 := by
sorry

#eval number_of_kids 13 7 28

end NUMINAMATH_CALUDE_kids_in_restaurant_group_l1084_108496


namespace NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l1084_108438

/-- Proves that Car A's speed is 58 mph given the problem conditions -/
theorem car_a_speed : ℝ → Prop :=
  fun (speed_a : ℝ) =>
    let initial_gap : ℝ := 10
    let overtake_distance : ℝ := 8
    let speed_b : ℝ := 50
    let time : ℝ := 2.25
    let distance_b : ℝ := speed_b * time
    let distance_a : ℝ := distance_b + initial_gap + overtake_distance
    speed_a = distance_a / time ∧ speed_a = 58

/-- The theorem is true -/
theorem car_a_speed_is_58 : ∃ (speed_a : ℝ), car_a_speed speed_a :=
sorry

end NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l1084_108438


namespace NUMINAMATH_CALUDE_max_value_of_f_l1084_108495

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 6 * x

-- Define the interval
def I : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 9/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1084_108495


namespace NUMINAMATH_CALUDE_peach_difference_l1084_108441

/-- Given information about peaches owned by Jake, Steven, and Jill -/
theorem peach_difference (jill steven jake : ℕ) 
  (h1 : jake = steven - 5)  -- Jake has 5 fewer peaches than Steven
  (h2 : steven = jill + 18) -- Steven has 18 more peaches than Jill
  (h3 : jill = 87)          -- Jill has 87 peaches
  : jake - jill = 13 :=     -- Prove that Jake has 13 more peaches than Jill
by sorry

end NUMINAMATH_CALUDE_peach_difference_l1084_108441


namespace NUMINAMATH_CALUDE_correct_base_notation_l1084_108402

def is_valid_base_representation (digits : List Nat) (base : Nat) : Prop :=
  digits.all (· < base) ∧ digits.head! > 0

theorem correct_base_notation :
  is_valid_base_representation [7, 5, 1] 9 ∧
  ¬is_valid_base_representation [7, 5, 1] 7 ∧
  ¬is_valid_base_representation [0, 9, 5] 12 ∧
  ¬is_valid_base_representation [9, 0, 1] 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_base_notation_l1084_108402


namespace NUMINAMATH_CALUDE_largest_common_divisor_fifteen_always_divides_l1084_108455

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def product (n : ℕ) : ℕ := n * (n+2) * (n+4) * (n+6) * (n+8)

theorem largest_common_divisor :
  ∀ (d : ℕ), d > 15 →
    ∃ (n : ℕ), is_odd n ∧ ¬(d ∣ product n) :=
sorry

theorem fifteen_always_divides :
  ∀ (n : ℕ), is_odd n → (15 ∣ product n) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_fifteen_always_divides_l1084_108455


namespace NUMINAMATH_CALUDE_sunlovers_always_happy_l1084_108403

theorem sunlovers_always_happy (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sunlovers_always_happy_l1084_108403


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1084_108458

/-- Two circles with radii R and r, where R and r are the roots of x^2 - 3x + 2 = 0,
    and whose centers are at a distance d = 3 apart, are externally tangent. -/
theorem circles_externally_tangent (R r : ℝ) (d : ℝ) : 
  (R^2 - 3*R + 2 = 0) → 
  (r^2 - 3*r + 2 = 0) → 
  (d = 3) → 
  (R + r = d) := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1084_108458


namespace NUMINAMATH_CALUDE_binomial_15_12_l1084_108492

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l1084_108492


namespace NUMINAMATH_CALUDE_brass_weight_l1084_108459

theorem brass_weight (copper_ratio : ℚ) (zinc_ratio : ℚ) (zinc_weight : ℚ) : 
  copper_ratio = 3 → 
  zinc_ratio = 7 → 
  zinc_weight = 70 → 
  (copper_ratio + zinc_ratio) * (zinc_weight / zinc_ratio) = 100 :=
by sorry

end NUMINAMATH_CALUDE_brass_weight_l1084_108459


namespace NUMINAMATH_CALUDE_range_of_m_l1084_108428

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x ∈ Set.Icc (1/2 : ℝ) 2, x^2 - 2*x - m ≤ 0) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1084_108428


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1084_108412

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -(1/2 : ℂ) * (1 + Complex.I)) : 
  z.im = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1084_108412


namespace NUMINAMATH_CALUDE_min_sum_odd_days_l1084_108400

/-- A sequence of 5 non-negative integers representing fish caught each day --/
def FishSequence := (ℕ × ℕ × ℕ × ℕ × ℕ)

/-- Check if a sequence is non-increasing --/
def is_non_increasing (seq : FishSequence) : Prop :=
  let (a, b, c, d, e) := seq
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e

/-- Calculate the sum of all elements in the sequence --/
def sum_sequence (seq : FishSequence) : ℕ :=
  let (a, b, c, d, e) := seq
  a + b + c + d + e

/-- Calculate the sum of 1st, 3rd, and 5th elements --/
def sum_odd_days (seq : FishSequence) : ℕ :=
  let (a, _, c, _, e) := seq
  a + c + e

/-- The main theorem --/
theorem min_sum_odd_days (seq : FishSequence) :
  is_non_increasing seq →
  sum_sequence seq = 100 →
  sum_odd_days seq ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_odd_days_l1084_108400


namespace NUMINAMATH_CALUDE_circle_center_l1084_108446

def is_circle (a : ℝ) : Prop :=
  ∃ (h : a^2 = a + 2 ∧ a^2 ≠ 0),
  ∀ (x y : ℝ), a^2*x^2 + (a+2)*y^2 + 4*x + 8*y + 5*a = 0 →
  ∃ (r : ℝ), (x + 2)^2 + (y + 4)^2 = r^2

theorem circle_center (a : ℝ) (h : is_circle a) :
  ∃ (x y : ℝ), a^2*x^2 + (a+2)*y^2 + 4*x + 8*y + 5*a = 0 ∧ x = -2 ∧ y = -4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l1084_108446


namespace NUMINAMATH_CALUDE_rhombus_all_sides_equal_rectangle_not_necessarily_l1084_108467

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- A rectangle is a quadrilateral with four right angles and opposite sides equal. -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Theorem stating that all sides of a rhombus are equal, but not necessarily for a rectangle -/
theorem rhombus_all_sides_equal_rectangle_not_necessarily (r : Rhombus) (rect : Rectangle) :
  (∀ (i j : Fin 4), r.sides i = r.sides j) ∧
  ¬(∀ (rect : Rectangle), rect.width = rect.height) :=
sorry

end NUMINAMATH_CALUDE_rhombus_all_sides_equal_rectangle_not_necessarily_l1084_108467


namespace NUMINAMATH_CALUDE_bahs_equivalent_to_1000_yahs_l1084_108420

/-- The number of bahs equivalent to one rah -/
def bah_per_rah : ℚ := 15 / 24

/-- The number of rahs equivalent to one yah -/
def rah_per_yah : ℚ := 9 / 15

/-- The number of bahs equivalent to 1000 yahs -/
def bahs_per_1000_yahs : ℚ := 1000 * rah_per_yah * bah_per_rah

theorem bahs_equivalent_to_1000_yahs : bahs_per_1000_yahs = 375 := by
  sorry

end NUMINAMATH_CALUDE_bahs_equivalent_to_1000_yahs_l1084_108420


namespace NUMINAMATH_CALUDE_function_properties_l1084_108434

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) (f'' : ℝ → ℝ) 
  (h_even : is_even f)
  (h_deriv : ∀ x, HasDerivAt f (f'' x) x)
  (h_eq : ∀ x, f (x - 1/2) + f (x + 1) = 0)
  (h_val : Real.exp 3 * f 2018 = 1)
  (h_ineq : ∀ x, f x > f'' (-x)) :
  {x : ℝ | f (x - 1) > 1 / Real.exp x} = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1084_108434


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1084_108452

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = 4 / 3) : 
  Real.sqrt (1 + (b / a)^2) = 5 / 3 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1084_108452


namespace NUMINAMATH_CALUDE_units_digit_of_72_cubed_minus_24_cubed_l1084_108436

theorem units_digit_of_72_cubed_minus_24_cubed : ∃ n : ℕ, 72^3 - 24^3 ≡ 4 [MOD 10] ∧ n * 10 + 4 = 72^3 - 24^3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_72_cubed_minus_24_cubed_l1084_108436


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1084_108447

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | x^2 + 2*x < 0}

theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {-2, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1084_108447


namespace NUMINAMATH_CALUDE_eggs_needed_for_scaled_cake_l1084_108445

/-- Represents the recipe for sponge cake -/
structure Recipe where
  eggs : ℝ
  flour : ℝ
  sugar : ℝ

/-- Calculates the total mass of the cake from a recipe -/
def totalMass (r : Recipe) : ℝ := r.eggs + r.flour + r.sugar

/-- The original recipe -/
def originalRecipe : Recipe := { eggs := 300, flour := 120, sugar := 100 }

/-- Theorem: The amount of eggs needed for 2600g of sponge cake is 1500g -/
theorem eggs_needed_for_scaled_cake (desiredMass : ℝ) 
  (h : desiredMass = 2600) : 
  (originalRecipe.eggs / totalMass originalRecipe) * desiredMass = 1500 := by
  sorry

end NUMINAMATH_CALUDE_eggs_needed_for_scaled_cake_l1084_108445


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_l1084_108425

theorem zoo_ticket_cost (adult_price : ℝ) : 
  (adult_price > 0) →
  (6 * adult_price + 5 * (adult_price / 2) + 3 * (adult_price - 1.5) = 40.5) →
  (10 * adult_price + 8 * (adult_price / 2) + 4 * (adult_price - 1.5) = 64.38) :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_l1084_108425


namespace NUMINAMATH_CALUDE_sum_9_equals_126_l1084_108468

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + a 8 = 15 + a 5

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a 1 + seq.a n)) / 2

/-- The theorem to be proved -/
theorem sum_9_equals_126 (seq : ArithmeticSequence) : sum_n seq 9 = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_9_equals_126_l1084_108468


namespace NUMINAMATH_CALUDE_helen_cookies_proof_l1084_108426

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 270

/-- The total number of cookies Helen baked till last night -/
def total_cookies : ℕ := 450

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_before_yesterday : ℕ := total_cookies - (cookies_yesterday + cookies_this_morning)

theorem helen_cookies_proof : 
  cookies_before_yesterday = 149 := by sorry

end NUMINAMATH_CALUDE_helen_cookies_proof_l1084_108426


namespace NUMINAMATH_CALUDE_parking_methods_count_l1084_108454

/-- Represents the number of parking spaces -/
def n : ℕ := 6

/-- Represents the number of cars to be parked -/
def k : ℕ := 3

/-- Calculates the number of ways to park cars when they are not adjacent -/
def non_adjacent_ways : ℕ := (n - k + 1).choose k * 2^k

/-- Calculates the number of ways to park cars when two are adjacent -/
def two_adjacent_ways : ℕ := 2 * k.choose 2 * (n - k).choose 1 * 2^2

/-- Calculates the number of ways to park cars when all are adjacent -/
def all_adjacent_ways : ℕ := (n - k + 1) * 2

/-- The total number of parking methods -/
def total_parking_methods : ℕ := non_adjacent_ways + two_adjacent_ways + all_adjacent_ways

theorem parking_methods_count : total_parking_methods = 528 := by sorry

end NUMINAMATH_CALUDE_parking_methods_count_l1084_108454


namespace NUMINAMATH_CALUDE_pebble_color_difference_l1084_108401

theorem pebble_color_difference (total_pebbles red_pebbles blue_pebbles : ℕ) 
  (h1 : total_pebbles = 40)
  (h2 : red_pebbles = 9)
  (h3 : blue_pebbles = 13)
  (h4 : (total_pebbles - red_pebbles - blue_pebbles) % 3 = 0) :
  blue_pebbles - (total_pebbles - red_pebbles - blue_pebbles) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_pebble_color_difference_l1084_108401


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l1084_108430

/-- The number of tan chips in the bag -/
def tan_chips : ℕ := 4

/-- The number of pink chips in the bag -/
def pink_chips : ℕ := 3

/-- The number of violet chips in the bag -/
def violet_chips : ℕ := 5

/-- The number of green chips in the bag -/
def green_chips : ℕ := 2

/-- The total number of chips in the bag -/
def total_chips : ℕ := tan_chips + pink_chips + violet_chips + green_chips

/-- The probability of drawing the chips as specified -/
def probability : ℚ := 1 / 42000

theorem chip_drawing_probability :
  (tan_chips.factorial * pink_chips.factorial * violet_chips.factorial * (3 + green_chips).factorial) / total_chips.factorial = probability := by
  sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l1084_108430


namespace NUMINAMATH_CALUDE_actual_distance_travelled_l1084_108463

/-- The actual distance travelled by a person, given two walking speeds and an additional distance condition. -/
theorem actual_distance_travelled (speed1 speed2 additional_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : additional_distance = 20)
  (h4 : (actual_distance / speed1) = ((actual_distance + additional_distance) / speed2)) :
  actual_distance = 50 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_travelled_l1084_108463


namespace NUMINAMATH_CALUDE_y_equals_zero_l1084_108460

theorem y_equals_zero (x y : ℝ) : (x + y)^5 - x^5 + y = 0 → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_zero_l1084_108460


namespace NUMINAMATH_CALUDE_thermostat_changes_l1084_108462

theorem thermostat_changes (initial_temp : ℝ) : 
  initial_temp = 40 →
  let doubled := initial_temp * 2
  let after_dad := doubled - 30
  let after_mom := after_dad * 0.7
  let final_temp := after_mom + 24
  final_temp = 59 := by sorry

end NUMINAMATH_CALUDE_thermostat_changes_l1084_108462


namespace NUMINAMATH_CALUDE_probability_complement_event_correct_l1084_108490

/-- The probability of event $\overline{A}$ occurring exactly $k$ times in $n$ trials, 
    given that the probability of event $A$ occurring in each trial is $P$. -/
def probability_complement_event (n k : ℕ) (P : ℝ) : ℝ :=
  (n.choose k) * (1 - P)^k * P^(n - k)

/-- Theorem stating that the probability of event $\overline{A}$ occurring exactly $k$ times 
    in $n$ trials, given that the probability of event $A$ occurring in each trial is $P$, 
    is equal to $C_n^k(1-P)^k P^{n-k}$. -/
theorem probability_complement_event_correct (n k : ℕ) (P : ℝ) 
    (h1 : 0 ≤ P) (h2 : P ≤ 1) (h3 : k ≤ n) : 
  probability_complement_event n k P = (n.choose k) * (1 - P)^k * P^(n - k) := by
  sorry

end NUMINAMATH_CALUDE_probability_complement_event_correct_l1084_108490


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l1084_108432

theorem min_value_of_sum_of_ratios (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (a + c) * (b + d) = a * c + b * d) : 
  a / b + b / c + c / d + d / a ≥ 8 ∧ 
  ∃ (a' b' c' d' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧
    (a' + c') * (b' + d') = a' * c' + b' * d' ∧
    a' / b' + b' / c' + c' / d' + d' / a' = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l1084_108432


namespace NUMINAMATH_CALUDE_equation_solution_l1084_108421

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 4 + 3 * Real.sqrt 19) ∧ (x₂ = 4 - 3 * Real.sqrt 19) ∧
  (∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1084_108421


namespace NUMINAMATH_CALUDE_positive_c_geq_one_l1084_108487

theorem positive_c_geq_one (a b : ℕ+) (c : ℝ) 
  (h_c_pos : c > 0) 
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : 
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_positive_c_geq_one_l1084_108487


namespace NUMINAMATH_CALUDE_trailer_count_proof_l1084_108414

theorem trailer_count_proof (initial_count : ℕ) (initial_avg_age : ℝ) (current_avg_age : ℝ) :
  initial_count = 30 ∧ initial_avg_age = 12 ∧ current_avg_age = 10 →
  ∃ (new_count : ℕ), 
    (initial_count * (initial_avg_age + 4) + new_count * 4) / (initial_count + new_count) = current_avg_age ∧
    new_count = 30 := by
  sorry

end NUMINAMATH_CALUDE_trailer_count_proof_l1084_108414


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1084_108417

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 9 * x^9 + 5 * x^8) + (2 * x^12 + x^10 + 2 * x^9 + 3 * x^8 + 4 * x^4 + 6 * x^2 + 9) =
  2 * x^12 + 13 * x^10 + 11 * x^9 + 8 * x^8 + 4 * x^4 + 6 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1084_108417


namespace NUMINAMATH_CALUDE_max_intersections_convex_polygons_l1084_108469

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  isConvex : Bool

/-- Represents the state of two polygons after rotation -/
structure RotatedPolygons (Q1 Q2 : ConvexPolygon) where
  canIntersect : Bool

/-- Calculates the maximum number of intersections between two rotated polygons -/
def maxIntersections (Q1 Q2 : ConvexPolygon) (state : RotatedPolygons Q1 Q2) : ℕ :=
  if state.canIntersect then Q1.sides * Q2.sides else 0

theorem max_intersections_convex_polygons :
  ∀ (Q1 Q2 : ConvexPolygon) (state : RotatedPolygons Q1 Q2),
    Q1.sides = 5 →
    Q2.sides = 7 →
    Q1.isConvex = true →
    Q2.isConvex = true →
    state.canIntersect = true →
    maxIntersections Q1 Q2 state = 35 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_convex_polygons_l1084_108469


namespace NUMINAMATH_CALUDE_total_photos_l1084_108405

def friends_photos : ℕ := 63
def family_photos : ℕ := 23

theorem total_photos : friends_photos + family_photos = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_photos_l1084_108405


namespace NUMINAMATH_CALUDE_max_value_parabola_l1084_108477

/-- The maximum value of y = -3x^2 + 7, where x is a real number, is 7. -/
theorem max_value_parabola :
  ∀ x : ℝ, -3 * x^2 + 7 ≤ 7 ∧ ∃ x₀ : ℝ, -3 * x₀^2 + 7 = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_parabola_l1084_108477


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l1084_108411

theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid_poured : ℝ)
  (final_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 50)
  (h5 : total_liquid_poured = 8)
  (h6 : final_vessel_capacity = 10)
  : (vessel1_capacity * vessel1_alcohol_percentage / 100 +
     vessel2_capacity * vessel2_alcohol_percentage / 100) /
    final_vessel_capacity * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l1084_108411


namespace NUMINAMATH_CALUDE_olivias_cookies_l1084_108410

/-- Proves the number of oatmeal cookies given the conditions of the problem -/
theorem olivias_cookies (cookies_per_baggie : ℕ) (total_baggies : ℕ) (chocolate_chip_cookies : ℕ)
  (h1 : cookies_per_baggie = 9)
  (h2 : total_baggies = 6)
  (h3 : chocolate_chip_cookies = 13) :
  cookies_per_baggie * total_baggies - chocolate_chip_cookies = 41 := by
  sorry

end NUMINAMATH_CALUDE_olivias_cookies_l1084_108410


namespace NUMINAMATH_CALUDE_correct_experimental_procedure_l1084_108499

-- Define the type for experimental procedures
inductive ExperimentalProcedure
| MicroorganismIsolation
| WineFermentation
| CellObservation
| EcoliCounting

-- Define the properties of each procedure
def requiresLight (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.MicroorganismIsolation => false
  | _ => true

def requiresOpenBottle (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.WineFermentation => false
  | _ => true

def adjustAperture (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.CellObservation => false
  | _ => true

def ensureDilution (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.EcoliCounting => true
  | _ => false

-- Theorem stating that E. coli counting is the correct procedure
theorem correct_experimental_procedure :
  ∀ p : ExperimentalProcedure,
    (p = ExperimentalProcedure.EcoliCounting) ↔
    (¬requiresLight p ∧ ¬requiresOpenBottle p ∧ ¬adjustAperture p ∧ ensureDilution p) :=
by sorry

end NUMINAMATH_CALUDE_correct_experimental_procedure_l1084_108499


namespace NUMINAMATH_CALUDE_union_of_sets_l1084_108435

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1084_108435


namespace NUMINAMATH_CALUDE_exponential_inequality_l1084_108443

theorem exponential_inequality (x a b : ℝ) 
  (h_x_pos : x > 0) 
  (h_ineq : 0 < b^x ∧ b^x < a^x ∧ a^x < 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) : 
  1 > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1084_108443


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1084_108422

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1084_108422


namespace NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l1084_108406

/-- Represents a cone-shaped mountain partially submerged in water -/
structure SubmergedMountain where
  totalHeight : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of a submerged mountain -/
def oceanDepth (mountain : SubmergedMountain) : ℝ :=
  mountain.totalHeight * (1 - (1 - mountain.aboveWaterVolumeFraction) ^ (1/3))

/-- Theorem stating that for a specific mountain configuration, the ocean depth is 648 feet -/
theorem ocean_depth_for_specific_mountain :
  let mountain : SubmergedMountain := {
    totalHeight := 12000,
    aboveWaterVolumeFraction := 1/6
  }
  oceanDepth mountain = 648 := by sorry

end NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l1084_108406


namespace NUMINAMATH_CALUDE_tree_growth_rate_l1084_108423

/-- Proves that a tree with given initial and final heights over a specific time period has a certain growth rate per week. -/
theorem tree_growth_rate 
  (initial_height : ℝ) 
  (final_height : ℝ) 
  (months : ℕ) 
  (weeks_per_month : ℕ) 
  (h1 : initial_height = 10)
  (h2 : final_height = 42)
  (h3 : months = 4)
  (h4 : weeks_per_month = 4) :
  (final_height - initial_height) / (months * weeks_per_month : ℝ) = 2 := by
  sorry

#check tree_growth_rate

end NUMINAMATH_CALUDE_tree_growth_rate_l1084_108423


namespace NUMINAMATH_CALUDE_min_distance_from_start_l1084_108484

/-- Represents a robot's movement on a 2D plane. -/
structure RobotMovement where
  /-- The distance the robot moves per minute. -/
  speed : ℝ
  /-- The total number of minutes the robot moves. -/
  total_time : ℕ
  /-- The number of minutes before the robot starts turning. -/
  initial_straight_time : ℕ

/-- Theorem stating the minimum distance from the starting point after the robot's movement. -/
theorem min_distance_from_start (r : RobotMovement) 
  (h1 : r.speed = 10)
  (h2 : r.total_time = 9)
  (h3 : r.initial_straight_time = 1) :
  ∃ (d : ℝ), d = 10 ∧ ∀ (final_pos : ℝ × ℝ), 
    (final_pos.1^2 + final_pos.2^2).sqrt ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_from_start_l1084_108484


namespace NUMINAMATH_CALUDE_least_n_with_property_l1084_108482

/-- The property that n^2 - n + 1 is divisible by some but not all k where 1 ≤ k ≤ n -/
def has_property (n : ℕ) : Prop :=
  ∃ k, 1 ≤ k ∧ k ≤ n ∧ (n^2 - n + 1) % k = 0 ∧
  ∃ m, 1 ≤ m ∧ m ≤ n ∧ (n^2 - n + 1) % m ≠ 0

/-- 5 is the least positive integer satisfying the property -/
theorem least_n_with_property :
  has_property 5 ∧ ∀ n, 0 < n → n < 5 → ¬(has_property n) :=
sorry

end NUMINAMATH_CALUDE_least_n_with_property_l1084_108482


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_l1084_108494

theorem rectangle_length_from_square (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_from_square_l1084_108494


namespace NUMINAMATH_CALUDE_money_theorem_l1084_108413

/-- The amount of money Fritz has -/
def fritz_money : ℕ := 40

/-- The amount of money Sean has -/
def sean_money : ℕ := fritz_money / 2 + 4

/-- The amount of money Rick has -/
def rick_money : ℕ := 3 * sean_money

/-- The amount of money Lindsey has -/
def lindsey_money : ℕ := 2 * (sean_money + rick_money)

/-- The total amount of money Lindsey, Rick, and Sean have combined -/
def total_money : ℕ := lindsey_money + rick_money + sean_money

theorem money_theorem : total_money = 288 := by
  sorry

end NUMINAMATH_CALUDE_money_theorem_l1084_108413


namespace NUMINAMATH_CALUDE_surface_area_is_14_l1084_108483

/-- The surface area of a rectangular prism formed by joining three 1x1x1 cubes side by side -/
def surface_area_of_prism : ℕ :=
  let length : ℕ := 3
  let width : ℕ := 1
  let height : ℕ := 1
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of the prism is 14 -/
theorem surface_area_is_14 : surface_area_of_prism = 14 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_is_14_l1084_108483


namespace NUMINAMATH_CALUDE_point_D_coordinates_l1084_108450

def vector_AB : ℝ × ℝ := (5, -3)
def point_C : ℝ × ℝ := (-1, 3)

theorem point_D_coordinates :
  ∀ D : ℝ × ℝ,
  (D.1 - point_C.1, D.2 - point_C.2) = (2 * vector_AB.1, 2 * vector_AB.2) →
  D = (9, -3) := by
sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l1084_108450


namespace NUMINAMATH_CALUDE_problem_1_l1084_108407

theorem problem_1 (x : ℝ) : (3*x + 1)*(3*x - 1) - (3*x + 1)^2 = -6*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1084_108407


namespace NUMINAMATH_CALUDE_evaluate_expression_l1084_108480

theorem evaluate_expression : Real.sqrt 5 * 5^(1/2 : ℝ) + 20 / 4 * 3 - 9^(3/2 : ℝ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1084_108480


namespace NUMINAMATH_CALUDE_correct_age_order_l1084_108431

-- Define the set of friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend
| George : Friend

-- Define a type for age comparisons
def AgeOrder := Friend → Friend → Prop

-- Define the problem conditions
def ProblemConditions (order : AgeOrder) : Prop :=
  -- All friends have different ages
  (∀ x y : Friend, x ≠ y → (order x y ∨ order y x)) ∧
  (∀ x y : Friend, order x y → ¬order y x) ∧
  -- Exactly one of the following statements is true
  (((order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    (¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    (∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    (∃ x : Friend, order x Friend.David)))

-- State the theorem
theorem correct_age_order (order : AgeOrder) :
  ProblemConditions order →
  (order Friend.David Friend.Emma ∧
   order Friend.Emma Friend.George ∧
   order Friend.George Friend.Fiona) :=
by sorry

end NUMINAMATH_CALUDE_correct_age_order_l1084_108431


namespace NUMINAMATH_CALUDE_total_evening_sales_l1084_108485

/-- Calculates the total evening sales given the conditions of the problem -/
theorem total_evening_sales :
  let remy_bottles : ℕ := 55
  let nick_bottles : ℕ := remy_bottles - 6
  let price_per_bottle : ℚ := 1/2
  let morning_sales : ℚ := (remy_bottles + nick_bottles : ℚ) * price_per_bottle
  let evening_sales : ℚ := morning_sales + 3
  evening_sales = 55 := by sorry

end NUMINAMATH_CALUDE_total_evening_sales_l1084_108485


namespace NUMINAMATH_CALUDE_selection_theorem_l1084_108461

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of representatives to be selected -/
def representatives : ℕ := 3

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

/-- The number of ways to select 3 representatives from 7 students,
    with the condition that only one of students A and B is selected -/
def selection_ways : ℕ := Nat.choose special_students 1 * Nat.choose (total_students - special_students) (representatives - 1)

theorem selection_theorem : selection_ways = 20 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l1084_108461


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1084_108418

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 3 * a + 5 * b = 47) 
  (eq2 : 4 * a + 2 * b = 38) : 
  a + b = 85 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1084_108418


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1084_108453

theorem possible_values_of_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → 
  (a = 6 ∨ a = 10) := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1084_108453


namespace NUMINAMATH_CALUDE_missing_mark_calculation_l1084_108479

def calculate_missing_mark (english math physics chemistry average : ℕ) : ℕ :=
  5 * average - (english + math + physics + chemistry)

theorem missing_mark_calculation (english math physics chemistry average biology : ℕ)
  (h1 : english = 76)
  (h2 : math = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 73)
  (h6 : biology = calculate_missing_mark english math physics chemistry average) :
  biology = 75 := by
  sorry

end NUMINAMATH_CALUDE_missing_mark_calculation_l1084_108479


namespace NUMINAMATH_CALUDE_jezebel_flower_cost_l1084_108449

/-- Calculates the total cost of flowers with discount and tax -/
def total_cost (rose_price : ℚ) (lily_price : ℚ) (sunflower_price : ℚ) (orchid_price : ℚ)
                (rose_quantity : ℕ) (lily_quantity : ℕ) (sunflower_quantity : ℕ) (orchid_quantity : ℕ)
                (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let rose_cost := rose_price * rose_quantity
  let lily_cost := lily_price * lily_quantity
  let sunflower_cost := sunflower_price * sunflower_quantity
  let orchid_cost := orchid_price * orchid_quantity
  let total_before_discount := rose_cost + lily_cost + sunflower_cost + orchid_cost
  let discount := discount_rate * (rose_cost + lily_cost)
  let total_after_discount := total_before_discount - discount
  let tax := tax_rate * total_after_discount
  total_after_discount + tax

/-- Theorem: The total cost of Jezebel's flower purchase is $142.90 -/
theorem jezebel_flower_cost :
  total_cost (3/2) (11/4) 3 (17/4) 24 14 8 10 (1/10) (7/100) = 1429/10 := by
  sorry


end NUMINAMATH_CALUDE_jezebel_flower_cost_l1084_108449


namespace NUMINAMATH_CALUDE_people_who_left_line_l1084_108493

theorem people_who_left_line (initial : ℕ) (joined : ℕ) (final : ℕ) : 
  initial = 7 → joined = 8 → final = 11 → initial - (initial - final + joined) = 4 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_line_l1084_108493


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1084_108444

theorem complex_equation_solution (z : ℂ) : Complex.I * (z - 1) = 1 + Complex.I * Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1084_108444


namespace NUMINAMATH_CALUDE_equal_cost_at_48_miles_l1084_108481

-- Define the daily rates and per-mile charges
def sunshine_daily_rate : ℝ := 17.99
def sunshine_per_mile : ℝ := 0.18
def city_daily_rate : ℝ := 18.95
def city_per_mile : ℝ := 0.16

-- Define the cost functions for each rental company
def sunshine_cost (miles : ℝ) : ℝ := sunshine_daily_rate + sunshine_per_mile * miles
def city_cost (miles : ℝ) : ℝ := city_daily_rate + city_per_mile * miles

-- Theorem stating that the costs are equal at 48 miles
theorem equal_cost_at_48_miles :
  sunshine_cost 48 = city_cost 48 := by sorry

end NUMINAMATH_CALUDE_equal_cost_at_48_miles_l1084_108481


namespace NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l1084_108465

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) →
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof is omitted
theorem bob_pennies_proof : ∃ a b : ℕ, bob_pennies a b := by
  sorry

end NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l1084_108465


namespace NUMINAMATH_CALUDE_N_rightmost_ten_l1084_108415

/-- A number with 1999 digits where each pair of consecutive digits
    is either a multiple of 17 or 23, and the sum of all digits is 9599 -/
def N : ℕ :=
  sorry

/-- Checks if a two-digit number is a multiple of 17 or 23 -/
def is_valid_pair (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- The property that each pair of consecutive digits in N
    is either a multiple of 17 or 23 -/
def valid_pairs (n : ℕ) : Prop :=
  ∀ i, i < 1998 → is_valid_pair ((n / 10^i) % 100)

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The rightmost ten digits of a natural number -/
def rightmost_ten (n : ℕ) : ℕ :=
  n % 10^10

theorem N_rightmost_ten :
  N ≥ 10^1998 ∧
  N < 10^1999 ∧
  valid_pairs N ∧
  digit_sum N = 9599 →
  rightmost_ten N = 3469234685 :=
sorry

end NUMINAMATH_CALUDE_N_rightmost_ten_l1084_108415


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1084_108471

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) : (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1084_108471


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1084_108442

def is_valid_pair (a b : ℂ) : Prop :=
  a^4 * b^7 = 1 ∧ a^8 * b^3 = 1

theorem count_valid_pairs :
  ∃! (n : ℕ), ∃ (S : Finset (ℂ × ℂ)),
    Finset.card S = n ∧
    (∀ (p : ℂ × ℂ), p ∈ S ↔ is_valid_pair p.1 p.2) ∧
    n = 16 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1084_108442


namespace NUMINAMATH_CALUDE_complex_number_location_l1084_108456

theorem complex_number_location :
  let z : ℂ := ((-1 : ℂ) + Complex.I) / ((1 : ℂ) + Complex.I) - 1
  z = -1 + Complex.I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1084_108456


namespace NUMINAMATH_CALUDE_remaining_balance_proof_l1084_108491

def gift_card_balance : ℝ := 100

def latte_price : ℝ := 3.75
def croissant_price : ℝ := 3.50
def bagel_price : ℝ := 2.25
def muffin_price : ℝ := 2.50
def special_drink_price : ℝ := 4.50
def cookie_price : ℝ := 1.25

def saturday_discount : ℝ := 0.10
def sunday_discount : ℝ := 0.20

def monday_expense : ℝ := latte_price + croissant_price + bagel_price
def tuesday_expense : ℝ := latte_price + croissant_price + muffin_price
def wednesday_expense : ℝ := latte_price + croissant_price + bagel_price
def thursday_expense : ℝ := latte_price + croissant_price + muffin_price
def friday_expense : ℝ := special_drink_price + croissant_price + bagel_price
def saturday_expense : ℝ := latte_price + croissant_price * (1 - saturday_discount)
def sunday_expense : ℝ := latte_price * (1 - sunday_discount) + croissant_price

def cookie_expense : ℝ := 5 * cookie_price

def total_expense : ℝ := monday_expense + tuesday_expense + wednesday_expense + thursday_expense + 
                         friday_expense + saturday_expense + sunday_expense + cookie_expense

theorem remaining_balance_proof : 
  gift_card_balance - total_expense = 31.60 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_proof_l1084_108491


namespace NUMINAMATH_CALUDE_work_ratio_women_to_men_l1084_108476

/-- The ratio of work done by women to men given specific work conditions -/
theorem work_ratio_women_to_men :
  let men_count : ℕ := 15
  let men_days : ℕ := 21
  let men_hours_per_day : ℕ := 8
  let women_count : ℕ := 21
  let women_days : ℕ := 36
  let women_hours_per_day : ℕ := 5
  let men_total_hours : ℕ := men_count * men_days * men_hours_per_day
  let women_total_hours : ℕ := women_count * women_days * women_hours_per_day
  (men_total_hours : ℚ) / women_total_hours = 2 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_work_ratio_women_to_men_l1084_108476


namespace NUMINAMATH_CALUDE_initial_books_eq_sold_plus_unsold_l1084_108466

/-- The number of books Ali had initially --/
def initial_books : ℕ := sorry

/-- The number of books Ali sold on Monday --/
def monday_sales : ℕ := 60

/-- The number of books Ali sold on Tuesday --/
def tuesday_sales : ℕ := 10

/-- The number of books Ali sold on Wednesday --/
def wednesday_sales : ℕ := 20

/-- The number of books Ali sold on Thursday --/
def thursday_sales : ℕ := 44

/-- The number of books Ali sold on Friday --/
def friday_sales : ℕ := 66

/-- The number of books not sold --/
def unsold_books : ℕ := 600

/-- Theorem stating that the initial number of books is equal to the sum of books sold on each day plus the number of books not sold --/
theorem initial_books_eq_sold_plus_unsold :
  initial_books = monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + unsold_books := by
  sorry

end NUMINAMATH_CALUDE_initial_books_eq_sold_plus_unsold_l1084_108466


namespace NUMINAMATH_CALUDE_goods_train_passing_time_goods_train_passing_time_approx_9_seconds_l1084_108474

/-- The time taken for a goods train to pass a man in another train -/
theorem goods_train_passing_time (passenger_train_speed goods_train_speed : ℝ) 
  (goods_train_length : ℝ) : ℝ :=
  let relative_speed := passenger_train_speed + goods_train_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  goods_train_length / relative_speed_mps

/-- Proof that the time taken is approximately 9 seconds -/
theorem goods_train_passing_time_approx_9_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |goods_train_passing_time 60 52 280 - 9| < ε :=
sorry

end NUMINAMATH_CALUDE_goods_train_passing_time_goods_train_passing_time_approx_9_seconds_l1084_108474


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1084_108472

theorem power_fraction_simplification :
  (3^2014 + 3^2012) / (3^2014 - 3^2012) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1084_108472


namespace NUMINAMATH_CALUDE_angle_problem_l1084_108404

theorem angle_problem (α β : Real) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h_distance : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 10 / 5)
  (h_tan : Real.tan (α/2) = 1/2) :
  (Real.cos (α - β) = 4/5) ∧
  (Real.cos α = 3/5) ∧
  (Real.cos β = 24/25) := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l1084_108404


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1084_108489

open Matrix

theorem matrix_equation_solution {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : IsUnit A) 
  (h_eq : (A - 3 • 1) * (A - 5 • 1) = -1) : 
  A + 10 • A⁻¹ = (6.5 : ℝ) • 1 := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1084_108489


namespace NUMINAMATH_CALUDE_complex_magnitude_l1084_108457

theorem complex_magnitude (z : ℂ) : (z + Complex.I) * (2 - Complex.I) = 11 + 7 * Complex.I → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1084_108457


namespace NUMINAMATH_CALUDE_triangular_pyramids_from_prism_l1084_108488

/-- The number of vertices in a triangular prism -/
def triangular_prism_vertices : ℕ := 6

/-- The number of vertices required to form a triangular pyramid -/
def triangular_pyramid_vertices : ℕ := 4

/-- The number of distinct triangular pyramids that can be formed using the vertices of a triangular prism -/
def distinct_triangular_pyramids : ℕ := 12

theorem triangular_pyramids_from_prism :
  distinct_triangular_pyramids = 12 :=
sorry

end NUMINAMATH_CALUDE_triangular_pyramids_from_prism_l1084_108488


namespace NUMINAMATH_CALUDE_stones_combine_l1084_108439

/-- Two natural numbers are similar if the larger is at most twice the smaller -/
def similar (a b : ℕ) : Prop := max a b ≤ 2 * min a b

/-- A step in the combining process -/
inductive CombineStep (n : ℕ)
  | combine (a b : ℕ) (h : a + b ≤ n) (hsim : similar a b) : CombineStep n

/-- A sequence of combining steps -/
def CombineSeq (n : ℕ) := List (CombineStep n)

/-- The result of applying a sequence of combining steps -/
def applySeq (n : ℕ) (seq : CombineSeq n) : List ℕ :=
  sorry

/-- The theorem stating that any number of single-stone piles can be combined into one pile -/
theorem stones_combine (n : ℕ) : 
  ∃ (seq : CombineSeq n), applySeq n seq = [n] :=
sorry

end NUMINAMATH_CALUDE_stones_combine_l1084_108439


namespace NUMINAMATH_CALUDE_pike_eel_fat_difference_l1084_108473

theorem pike_eel_fat_difference (herring_fat eel_fat : ℕ) (pike_fat : ℕ) 
  (fish_count : ℕ) (total_fat : ℕ) : 
  herring_fat = 40 →
  eel_fat = 20 →
  pike_fat > eel_fat →
  fish_count = 40 →
  fish_count * herring_fat + fish_count * eel_fat + fish_count * pike_fat = total_fat →
  total_fat = 3600 →
  pike_fat - eel_fat = 10 := by
sorry

end NUMINAMATH_CALUDE_pike_eel_fat_difference_l1084_108473


namespace NUMINAMATH_CALUDE_susan_hourly_rate_l1084_108427

/-- Susan's vacation and pay structure -/
structure VacationPay where
  work_days_per_week : ℕ
  paid_vacation_days : ℕ
  hours_per_day : ℕ
  missed_pay : ℕ
  vacation_weeks : ℕ

/-- Calculate Susan's hourly pay rate -/
def hourly_pay_rate (v : VacationPay) : ℚ :=
  let total_vacation_days := v.vacation_weeks * v.work_days_per_week
  let unpaid_vacation_days := total_vacation_days - v.paid_vacation_days
  let daily_pay := v.missed_pay / unpaid_vacation_days
  daily_pay / v.hours_per_day

/-- Theorem: Susan's hourly pay rate is $15 -/
theorem susan_hourly_rate :
  let v : VacationPay := {
    work_days_per_week := 5,
    paid_vacation_days := 6,
    hours_per_day := 8,
    missed_pay := 480,
    vacation_weeks := 2
  }
  hourly_pay_rate v = 15 := by sorry

end NUMINAMATH_CALUDE_susan_hourly_rate_l1084_108427


namespace NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1084_108470

/-- The sum of the interior angle of a regular pentagon and the interior angle of a regular triangle is 168°. -/
theorem pentagon_triangle_angle_sum : 
  let pentagon_angle : ℝ := 180 * (5 - 2) / 5
  let triangle_angle : ℝ := 180 * (3 - 2) / 3
  pentagon_angle + triangle_angle = 168 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1084_108470


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l1084_108408

/-- The Razorback t-shirt shop problem -/
theorem razorback_tshirt_sales (price : ℕ) (arkansas_sales : ℕ) (texas_tech_revenue : ℕ) :
  price = 78 →
  arkansas_sales = 172 →
  texas_tech_revenue = 1092 →
  arkansas_sales + (texas_tech_revenue / price) = 186 :=
by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l1084_108408


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l1084_108478

theorem quadratic_equation_integer_solutions :
  ∀ (x n : ℤ), x^2 + 3*x + 9 - 9*n^2 = 0 → (x = 0 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l1084_108478


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1084_108429

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the lines
def l (k x y : ℝ) : Prop := y = k * x + 1
def l₁ (k₁ x y : ℝ) : Prop := y = k₁ * x + 1

-- Define the symmetry line
def sym_line (x y : ℝ) : Prop := y = x + 1

-- Define the theorem
theorem ellipse_line_intersection
  (k k₁ : ℝ) 
  (hk : k > 0) 
  (hk_neq : k ≠ 1) 
  (h_sym : ∀ x y, l k x y ↔ l₁ k₁ (y - 1) (x - 1)) :
  ∃ P : ℝ × ℝ, 
    k * k₁ = 1 ∧ 
    ∀ M N : ℝ × ℝ, 
      (E M.1 M.2 ∧ l k M.1 M.2) → 
      (E N.1 N.2 ∧ l₁ k₁ N.1 N.2) → 
      ∃ t : ℝ, P = (1 - t) • M + t • N :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1084_108429


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1084_108464

theorem arithmetic_sequence_problem (n : ℕ) (sum : ℝ) (min_term max_term : ℝ) :
  n = 300 ∧ 
  sum = 22500 ∧ 
  min_term = 5 ∧ 
  max_term = 150 →
  let avg : ℝ := sum / n
  let d : ℝ := min ((avg - min_term) / (n - 1)) ((max_term - avg) / (n - 1))
  let L : ℝ := avg - (75 - 1) * d
  let G : ℝ := avg + (75 - 1) * d
  G - L = 31500 / 299 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1084_108464


namespace NUMINAMATH_CALUDE_number_comparisons_l1084_108475

theorem number_comparisons :
  (0.5 < 0.8) ∧ (0.5 < 0.7) ∧ (Real.log 125 < Real.log 1215) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l1084_108475


namespace NUMINAMATH_CALUDE_probability_blue_red_white_l1084_108498

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  blue : ℕ
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a specific sequence of marbles -/
def probability_of_sequence (counts : MarbleCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.white
  (counts.blue : ℚ) / total *
  (counts.red : ℚ) / (total - 1) *
  (counts.white : ℚ) / (total - 2)

/-- The main theorem stating the probability of drawing blue, red, then white -/
theorem probability_blue_red_white :
  probability_of_sequence ⟨4, 3, 6⟩ = 6 / 143 := by
  sorry

end NUMINAMATH_CALUDE_probability_blue_red_white_l1084_108498


namespace NUMINAMATH_CALUDE_power_division_equals_integer_l1084_108437

theorem power_division_equals_integer : 3^18 / 27^3 = 19683 := by sorry

end NUMINAMATH_CALUDE_power_division_equals_integer_l1084_108437


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l1084_108424

theorem fraction_sum_theorem (a b c : ℝ) (h : ((a - b) * (b - c) * (c - a)) / ((a + b) * (b + c) * (c + a)) = 2004 / 2005) :
  a / (a + b) + b / (b + c) + c / (c + a) = 4011 / 4010 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l1084_108424


namespace NUMINAMATH_CALUDE_factorization_equality_l1084_108486

theorem factorization_equality (x y : ℝ) : x^3*y - x*y = x*y*(x - 1)*(x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1084_108486


namespace NUMINAMATH_CALUDE_loan_period_is_three_l1084_108409

/-- The period of a loan (in years) where:
  - A lends Rs. 3500 to B at 10% per annum
  - B lends Rs. 3500 to C at 11.5% per annum
  - B gains Rs. 157.5 -/
def loanPeriod : ℝ → Prop := λ T =>
  let principal : ℝ := 3500
  let rateAtoB : ℝ := 10
  let rateBtoC : ℝ := 11.5
  let bGain : ℝ := 157.5
  let interestAtoB : ℝ := principal * rateAtoB * T / 100
  let interestBtoC : ℝ := principal * rateBtoC * T / 100
  interestBtoC - interestAtoB = bGain

theorem loan_period_is_three : loanPeriod 3 := by
  sorry

end NUMINAMATH_CALUDE_loan_period_is_three_l1084_108409


namespace NUMINAMATH_CALUDE_petyas_race_time_l1084_108440

/-- Proves that Petya's actual time is greater than the planned time -/
theorem petyas_race_time (a V : ℝ) (h1 : a > 0) (h2 : V > 0) :
  a / V < a / (2.5 * V) + a / (1.6 * V) :=
by sorry

end NUMINAMATH_CALUDE_petyas_race_time_l1084_108440
