import Mathlib

namespace quadratic_real_roots_range_l3734_373457

/-- The range of values for the real number a such that at least one of the given quadratic equations has real roots -/
theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + a^2 = 0 ∨ x^2 + 2*a*x - 2*a = 0) ↔ 
  a ≤ -2 ∨ (-1/3 ≤ a ∧ a < 1) ∨ 0 ≤ a := by
sorry

end quadratic_real_roots_range_l3734_373457


namespace rihanna_shopping_theorem_l3734_373491

def calculate_remaining_money (initial_amount : ℕ) (mango_count : ℕ) (juice_count : ℕ) (mango_price : ℕ) (juice_price : ℕ) : ℕ :=
  initial_amount - (mango_count * mango_price + juice_count * juice_price)

theorem rihanna_shopping_theorem (initial_amount : ℕ) (mango_count : ℕ) (juice_count : ℕ) (mango_price : ℕ) (juice_price : ℕ) :
  calculate_remaining_money initial_amount mango_count juice_count mango_price juice_price =
  initial_amount - (mango_count * mango_price + juice_count * juice_price) :=
by
  sorry

#eval calculate_remaining_money 50 6 6 3 3

end rihanna_shopping_theorem_l3734_373491


namespace imaginary_power_sum_l3734_373442

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^23 + (i^105 * i^17) = -i - 1 := by sorry

end imaginary_power_sum_l3734_373442


namespace ellipse_equation_constants_l3734_373490

def ellipse_constants (f1 f2 p : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem ellipse_equation_constants :
  let f1 : ℝ × ℝ := (3, 1)
  let f2 : ℝ × ℝ := (3, 7)
  let p : ℝ × ℝ := (12, 2)
  let (a, b, h, k) := ellipse_constants f1 f2 p
  (a = (Real.sqrt 82 + Real.sqrt 106) / 2) ∧
  (b = Real.sqrt ((Real.sqrt 82 + Real.sqrt 106)^2 / 4 - 9)) ∧
  (h = 3) ∧
  (k = 4) :=
by sorry

end ellipse_equation_constants_l3734_373490


namespace max_value_of_f_l3734_373485

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c) ∧
  f c = 2 := by
  sorry

end max_value_of_f_l3734_373485


namespace quadratic_equation_proof_l3734_373422

theorem quadratic_equation_proof (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 - 4*x1 - 2*m + 5 = 0 ∧ 
    x2^2 - 4*x2 - 2*m + 5 = 0 ∧
    x1*x2 + x1 + x2 = m^2 + 6) →
  m = 1 :=
by sorry

end quadratic_equation_proof_l3734_373422


namespace smallest_prime_after_six_nonprimes_l3734_373495

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def has_six_consecutive_nonprimes (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ 
    ∀ i : ℕ, i ≥ k ∧ i < k + 6 → ¬(is_prime i)

theorem smallest_prime_after_six_nonprimes :
  (is_prime 97) ∧ 
  (has_six_consecutive_nonprimes 96) ∧ 
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ has_six_consecutive_nonprimes (p - 1))) :=
sorry

end smallest_prime_after_six_nonprimes_l3734_373495


namespace no_positive_integer_solutions_l3734_373469

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0 :=
by sorry

end no_positive_integer_solutions_l3734_373469


namespace solution_set_rational_inequality_l3734_373413

theorem solution_set_rational_inequality :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end solution_set_rational_inequality_l3734_373413


namespace largest_angle_ABC_l3734_373410

theorem largest_angle_ABC (AC BC : ℝ) (angle_BAC : ℝ) : 
  AC = 5 * Real.sqrt 2 →
  BC = 5 →
  angle_BAC = 30 * π / 180 →
  ∃ (angle_ABC : ℝ), 
    angle_ABC ≤ 135 * π / 180 ∧
    ∀ (other_angle_ABC : ℝ), 
      (AC / Real.sin angle_BAC = BC / Real.sin other_angle_ABC) →
      other_angle_ABC ≤ angle_ABC := by
sorry

end largest_angle_ABC_l3734_373410


namespace slow_train_speed_l3734_373416

-- Define the problem parameters
def total_distance : ℝ := 901
def fast_train_speed : ℝ := 58
def slow_train_departure_time : ℝ := 5.5  -- 5:30 AM in decimal hours
def fast_train_departure_time : ℝ := 9.5  -- 9:30 AM in decimal hours
def meeting_time : ℝ := 16.5  -- 4:30 PM in decimal hours

-- Define the theorem
theorem slow_train_speed :
  let slow_train_travel_time : ℝ := meeting_time - slow_train_departure_time
  let fast_train_travel_time : ℝ := meeting_time - fast_train_departure_time
  let fast_train_distance : ℝ := fast_train_speed * fast_train_travel_time
  let slow_train_distance : ℝ := total_distance - fast_train_distance
  slow_train_distance / slow_train_travel_time = 45 := by
  sorry


end slow_train_speed_l3734_373416


namespace exponent_division_l3734_373418

theorem exponent_division (x : ℝ) (hx : x ≠ 0) : 2 * x^4 / x^3 = 2 * x := by
  sorry

end exponent_division_l3734_373418


namespace hex_tile_difference_specific_hex_tile_difference_l3734_373496

/-- Represents a hexagonal tile arrangement with blue and green tiles -/
structure HexTileArrangement where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of hexagonal tiles around an existing arrangement -/
def add_border (arrangement : HexTileArrangement) (border_color : String) : HexTileArrangement :=
  match border_color with
  | "green" => { blue_tiles := arrangement.blue_tiles, 
                 green_tiles := arrangement.green_tiles + (arrangement.blue_tiles + arrangement.green_tiles) / 2 }
  | "blue" => { blue_tiles := arrangement.blue_tiles + (arrangement.blue_tiles + arrangement.green_tiles) / 2 + 3, 
                green_tiles := arrangement.green_tiles }
  | _ => arrangement

/-- The main theorem stating the difference in tile counts after adding two borders -/
theorem hex_tile_difference (initial : HexTileArrangement) :
  let with_green_border := add_border initial "green"
  let final := add_border with_green_border "blue"
  final.blue_tiles - final.green_tiles = 16 :=
by
  sorry

/-- The specific instance of the hexagonal tile arrangement -/
def initial_arrangement : HexTileArrangement := { blue_tiles := 20, green_tiles := 10 }

/-- Applying the theorem to the specific instance -/
theorem specific_hex_tile_difference :
  let with_green_border := add_border initial_arrangement "green"
  let final := add_border with_green_border "blue"
  final.blue_tiles - final.green_tiles = 16 :=
by
  sorry

end hex_tile_difference_specific_hex_tile_difference_l3734_373496


namespace sum_of_interior_angles_tenth_polygon_l3734_373480

/-- The number of sides of the nth polygon in the sequence -/
def sides (n : ℕ) : ℕ := n + 2

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The 10th polygon in the sequence -/
def tenth_polygon : ℕ := 10

theorem sum_of_interior_angles_tenth_polygon :
  interior_angle_sum (sides tenth_polygon) = 1440 := by
  sorry

end sum_of_interior_angles_tenth_polygon_l3734_373480


namespace exists_non_increasing_f_l3734_373440

theorem exists_non_increasing_f :
  ∃ a : ℝ, a < 0 ∧
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
  let f := fun x => a * x + Real.log x
  f x₁ ≥ f x₂ :=
sorry

end exists_non_increasing_f_l3734_373440


namespace exists_divisible_by_11_in_39_consecutive_integers_l3734_373474

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_divisible_by_11_in_39_consecutive_integers :
  ∀ (start : ℕ), ∃ (k : ℕ), k ∈ Finset.range 39 ∧ (sumOfDigits (start + k) % 11 = 0) := by
  sorry

end exists_divisible_by_11_in_39_consecutive_integers_l3734_373474


namespace geometric_sequence_a4_l3734_373494

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  is_geometric a → a 1 = 8 → a 2 * a 3 = -8 → a 4 = -1 :=
by
  sorry

end geometric_sequence_a4_l3734_373494


namespace sum_of_reciprocals_l3734_373408

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 4*x*y) : 
  1/x + 1/y = 1 := by
sorry

end sum_of_reciprocals_l3734_373408


namespace min_sum_fraction_min_sum_fraction_achievable_l3734_373482

theorem min_sum_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 1 / (2 * Real.rpow 2 (1/3)) :=
sorry

theorem min_sum_fraction_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 1 / (2 * Real.rpow 2 (1/3)) :=
sorry

end min_sum_fraction_min_sum_fraction_achievable_l3734_373482


namespace sum_of_cubes_theorem_l3734_373489

theorem sum_of_cubes_theorem (a b : ℤ) : 
  a * b = 12 → a^3 + b^3 = 91 → a^3 + b^3 = 91 := by
  sorry

end sum_of_cubes_theorem_l3734_373489


namespace inscribed_circle_radius_l3734_373409

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- Assumption that the hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0
  /-- Assumption that the area is positive -/
  area_pos : area > 0
  /-- Assumption that the radius is positive -/
  radius_pos : radius > 0

/-- Theorem stating that for a right-angled triangle with hypotenuse 9 and area 36,
    the radius of the inscribed circle is 3 -/
theorem inscribed_circle_radius
  (triangle : RightTriangleWithInscribedCircle)
  (h1 : triangle.hypotenuse = 9)
  (h2 : triangle.area = 36) :
  triangle.radius = 3 := by
  sorry

end inscribed_circle_radius_l3734_373409


namespace complex_magnitude_example_l3734_373479

theorem complex_magnitude_example : Complex.abs (-3 + (8/5)*Complex.I) = 17/5 := by
  sorry

end complex_magnitude_example_l3734_373479


namespace sum_divisors_400_has_one_prime_factor_l3734_373446

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive divisors of 400 has exactly one distinct prime factor -/
theorem sum_divisors_400_has_one_prime_factor :
  num_distinct_prime_factors (sum_of_divisors 400) = 1 := by sorry

end sum_divisors_400_has_one_prime_factor_l3734_373446


namespace parallelogram_height_relation_crosswalk_problem_l3734_373452

/-- Given a parallelogram with sides a, b, height h_a perpendicular to side a,
    prove that the height h_b perpendicular to side b is (a * h_a) / b -/
theorem parallelogram_height_relation (a b h_a h_b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hha : h_a > 0) (hhb : h_b > 0) :
  a * h_a = b * h_b :=
by sorry

/-- Prove that for a parallelogram with a = 25, b = 50, and h_a = 60,
    the height h_b perpendicular to side b is 30 -/
theorem crosswalk_problem (a b h_a h_b : ℝ) 
    (ha : a = 25) (hb : b = 50) (hha : h_a = 60) :
  h_b = 30 :=
by sorry

end parallelogram_height_relation_crosswalk_problem_l3734_373452


namespace probability_same_tribe_l3734_373404

def total_participants : ℕ := 18
def tribe_size : ℕ := 9
def num_quitters : ℕ := 3

theorem probability_same_tribe :
  (Nat.choose tribe_size num_quitters * 2 : ℚ) / Nat.choose total_participants num_quitters = 7 / 34 := by
  sorry

end probability_same_tribe_l3734_373404


namespace weakly_increasing_g_implies_m_eq_4_l3734_373425

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + (4-m)*x + m

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Define what it means for a function to be decreasing on an interval
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- Define what it means for a function to be weakly increasing on an interval
def IsWeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  IsIncreasing f a b ∧ IsDecreasing (fun x => f x / x) a b

-- State the theorem
theorem weakly_increasing_g_implies_m_eq_4 :
  ∀ m : ℝ, IsWeaklyIncreasing (g m) 0 2 → m = 4 :=
by sorry

end weakly_increasing_g_implies_m_eq_4_l3734_373425


namespace solve_equation_and_evaluate_l3734_373493

theorem solve_equation_and_evaluate : ∃ x : ℝ, 
  (5 * x - 3 = 15 * x + 15) ∧ (6 * (x + 5) = 19.2) := by
  sorry

end solve_equation_and_evaluate_l3734_373493


namespace complex_equation_solution_l3734_373455

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z = 1 + Complex.I → z = Complex.I := by
  sorry

end complex_equation_solution_l3734_373455


namespace cos_pi_half_minus_two_alpha_l3734_373433

theorem cos_pi_half_minus_two_alpha (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 := by
  sorry

end cos_pi_half_minus_two_alpha_l3734_373433


namespace buffet_meal_combinations_l3734_373439

theorem buffet_meal_combinations : 
  let meat_options : ℕ := 3
  let vegetable_options : ℕ := 5
  let dessert_options : ℕ := 5
  let meat_selections : ℕ := 1
  let vegetable_selections : ℕ := 3
  let dessert_selections : ℕ := 2
  (meat_options.choose meat_selections) * 
  (vegetable_options.choose vegetable_selections) * 
  (dessert_options.choose dessert_selections) = 300 := by
sorry

end buffet_meal_combinations_l3734_373439


namespace first_part_speed_l3734_373460

/-- Proves that given a total trip distance of 255 miles, with the second part being 3 hours at 55 mph,
    the speed S for the first 2 hours must be 45 mph. -/
theorem first_part_speed (total_distance : ℝ) (first_duration : ℝ) (second_duration : ℝ) (second_speed : ℝ) :
  total_distance = 255 →
  first_duration = 2 →
  second_duration = 3 →
  second_speed = 55 →
  ∃ S : ℝ, S = 45 ∧ total_distance = first_duration * S + second_duration * second_speed :=
by sorry

end first_part_speed_l3734_373460


namespace absolute_value_inequality_l3734_373438

theorem absolute_value_inequality (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end absolute_value_inequality_l3734_373438


namespace boys_to_girls_ratio_l3734_373430

theorem boys_to_girls_ratio (S : ℕ) (G : ℕ) (h1 : S > 0) (h2 : G > 0) 
  (h3 : 2 * G = 3 * (S / 5)) : 
  (S - G : ℚ) / G = 7 / 3 := by sorry

end boys_to_girls_ratio_l3734_373430


namespace divisor_problem_l3734_373432

theorem divisor_problem (k : ℕ) : 12^k ∣ 856736 → k = 0 := by
  sorry

end divisor_problem_l3734_373432


namespace division_problem_l3734_373401

theorem division_problem (number : ℕ) : 
  (number / 179 = 89) ∧ (number % 179 = 37) → number = 15968 := by
  sorry

end division_problem_l3734_373401


namespace abc_sum_range_l3734_373435

theorem abc_sum_range (a b c : ℝ) (h : a + b + 2*c = 0) :
  ∃ y : ℝ, y ≤ 0 ∧ a*b + a*c + b*c = y ∧
  ∀ z : ℝ, z ≤ 0 → ∃ a' b' c' : ℝ, a' + b' + 2*c' = 0 ∧ a'*b' + a'*c' + b'*c' = z :=
by sorry

end abc_sum_range_l3734_373435


namespace find_adult_ticket_cost_l3734_373415

def adult_ticket_cost (total_cost children_cost : ℕ) : Prop :=
  ∃ (adult_cost : ℕ), adult_cost + 6 * children_cost = total_cost

theorem find_adult_ticket_cost :
  adult_ticket_cost 155 20 → ∃ (adult_cost : ℕ), adult_cost = 35 :=
by
  sorry

end find_adult_ticket_cost_l3734_373415


namespace total_accessories_is_712_l3734_373431

/-- Calculates the total number of accessories used by Jane and Emily for their dresses -/
def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_accessories_per_dress := 3 + 2 + 1 + 4
  let emily_accessories_per_dress := 2 + 3 + 2 + 5 + 1
  jane_dresses * jane_accessories_per_dress + emily_dresses * emily_accessories_per_dress

/-- Theorem stating that the total number of accessories is 712 -/
theorem total_accessories_is_712 : total_accessories = 712 := by
  sorry

end total_accessories_is_712_l3734_373431


namespace max_min_x2_xy_y2_l3734_373454

theorem max_min_x2_xy_y2 (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (A_min A_max : ℝ), A_min = 2 ∧ A_max = 6 ∧
  ∀ A, A = x^2 + x*y + y^2 → A_min ≤ A ∧ A ≤ A_max :=
by sorry

end max_min_x2_xy_y2_l3734_373454


namespace f_strictly_increasing_l3734_373471

def f (x : ℝ) : ℝ := x^3 + x

theorem f_strictly_increasing : StrictMono f := by sorry

end f_strictly_increasing_l3734_373471


namespace round_1647_to_hundredth_l3734_373400

def round_to_hundredth (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

theorem round_1647_to_hundredth :
  round_to_hundredth (1647 / 1000) = 165 / 100 := by
  sorry

end round_1647_to_hundredth_l3734_373400


namespace inequality_solution_set_l3734_373445

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3/5) 1 := by
  sorry

end inequality_solution_set_l3734_373445


namespace units_digit_of_p_plus_two_l3734_373402

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_p_plus_two (p : ℕ) 
  (h1 : is_positive_even p)
  (h2 : has_positive_units_digit p)
  (h3 : units_digit (p^3) - units_digit (p^2) = 0) :
  units_digit (p + 2) = 8 := by
  sorry

end units_digit_of_p_plus_two_l3734_373402


namespace sqrt_meaningful_range_l3734_373472

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) ↔ x ≥ 5 := by sorry

end sqrt_meaningful_range_l3734_373472


namespace trajectory_of_G_l3734_373462

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + Real.sqrt 7)^2 + y^2 = 64

-- Define the fixed point N
def point_N : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define a point P on the circle M
def point_P (x y : ℝ) : Prop := circle_M x y

-- Define point Q on line NP
def point_Q (x y : ℝ) : Prop := ∃ t : ℝ, (x, y) = ((1 - t) * point_N.1 + t * x, (1 - t) * point_N.2 + t * y)

-- Define point G on line segment MP
def point_G (x y : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (x, y) = ((1 - t) * (-Real.sqrt 7) + t * x, t * y)

-- Define the condition NP = 2NQ
def condition_NP_2NQ (x_p y_p x_q y_q : ℝ) : Prop :=
  (x_p - point_N.1, y_p - point_N.2) = (2 * (x_q - point_N.1), 2 * (y_q - point_N.2))

-- Define the condition GQ ⋅ NP = 0
def condition_GQ_perp_NP (x_g y_g x_q y_q x_p y_p : ℝ) : Prop :=
  (x_g - x_q) * (x_p - point_N.1) + (y_g - y_q) * (y_p - point_N.2) = 0

theorem trajectory_of_G (x y : ℝ) :
  (∃ x_p y_p x_q y_q, 
    point_P x_p y_p ∧
    point_Q x_q y_q ∧
    point_G x y ∧
    condition_NP_2NQ x_p y_p x_q y_q ∧
    condition_GQ_perp_NP x y x_q y_q x_p y_p) →
  x^2/16 + y^2/9 = 1 :=
sorry

end trajectory_of_G_l3734_373462


namespace cookie_theorem_l3734_373456

def cookie_problem (initial_cookies eaten_cookies given_cookies : ℕ) : Prop :=
  initial_cookies = eaten_cookies + given_cookies ∧
  eaten_cookies - given_cookies = 11

theorem cookie_theorem :
  cookie_problem 17 14 3 := by
  sorry

end cookie_theorem_l3734_373456


namespace one_fifth_of_seven_times_nine_l3734_373463

theorem one_fifth_of_seven_times_nine : (1 / 5 : ℚ) * (7 * 9) = 12.6 := by
  sorry

end one_fifth_of_seven_times_nine_l3734_373463


namespace problem_1_problem_2_l3734_373476

-- Problem 1
theorem problem_1 : (-3)^2 + (Real.pi - 1/2)^0 - |(-4)| = 6 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) : 
  (1 - 1/(a+1)) * ((a^2 + 2*a + 1)/a) = a + 1 := by sorry

end problem_1_problem_2_l3734_373476


namespace complex_number_quadrant_l3734_373419

theorem complex_number_quadrant : ∃ (z : ℂ), z = (4 * Complex.I) / (1 + Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end complex_number_quadrant_l3734_373419


namespace correct_calculation_l3734_373407

theorem correct_calculation : ∃! x : ℤ, (2 - 3 = x ∧ x = -1) ∧
  ¬((-3)^2 = -9) ∧
  ¬(-3^2 = -6) ∧
  ¬(-3 - (-2) = -5) := by
  sorry

end correct_calculation_l3734_373407


namespace power_of_five_divides_l3734_373487

/-- Sequence of positive integers defined recursively -/
def x : ℕ → ℕ
  | 0 => 2  -- We use 0-based indexing in Lean
  | n + 1 => 2 * (x n)^3 + x n

/-- The statement to be proved -/
theorem power_of_five_divides (n : ℕ) : 
  ∃ k : ℕ, x n^2 + 1 = 5^(n+1) * k ∧ ¬(5 ∣ k) := by
  sorry

end power_of_five_divides_l3734_373487


namespace distinct_committees_l3734_373478

/-- The number of teams in the volleyball league -/
def numTeams : ℕ := 5

/-- The number of players in each team -/
def playersPerTeam : ℕ := 8

/-- The number of committee members selected from the host team -/
def hostCommitteeMembers : ℕ := 4

/-- The number of committee members selected from each non-host team -/
def nonHostCommitteeMembers : ℕ := 1

/-- The total number of distinct tournament committees over one complete rotation -/
def totalCommittees : ℕ := numTeams * (Nat.choose playersPerTeam hostCommitteeMembers) * (playersPerTeam ^ (numTeams - 1))

theorem distinct_committees :
  totalCommittees = 1433600 := by
  sorry

end distinct_committees_l3734_373478


namespace airline_capacity_proof_l3734_373420

/-- Calculates the number of passengers an airline can accommodate daily -/
def airline_capacity (num_airplanes : ℕ) (rows_per_airplane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) : ℕ :=
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day

/-- Proves that the airline company can accommodate 1400 passengers daily -/
theorem airline_capacity_proof :
  airline_capacity 5 20 7 2 = 1400 := by
  sorry

#eval airline_capacity 5 20 7 2

end airline_capacity_proof_l3734_373420


namespace triangle_angle_theorem_l3734_373426

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem triangle_angle_theorem 
  (A B C : ℝ × ℝ) 
  (E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_bac : angle (B - A) (C - A) = 30 * π / 180)
  (h_e_on_bc : PointOnSegment E B C)
  (h_be_ec : 3 * ‖E - B‖ = 2 * ‖C - E‖)
  (h_eab : angle (E - A) (B - A) = 45 * π / 180) :
  angle (A - C) (B - C) = 15 * π / 180 :=
sorry

end triangle_angle_theorem_l3734_373426


namespace regular_polygon_angle_relation_l3734_373414

theorem regular_polygon_angle_relation : 
  ∀ n : ℕ, 
  n ≥ 3 →
  (360 / n : ℚ) = (120 / 5 : ℚ) →
  n = 15 := by
sorry

end regular_polygon_angle_relation_l3734_373414


namespace opposite_numbers_l3734_373434

theorem opposite_numbers : -3 = -(Real.sqrt ((-3)^2)) := by sorry

end opposite_numbers_l3734_373434


namespace root_sum_reciprocal_l3734_373475

theorem root_sum_reciprocal (p q r : ℝ) (A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x^3 - 23*x^2 + 85*x - 72 = (x - p)*(x - q)*(x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 23*s^2 + 85*s - 72) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 248 :=
by sorry

end root_sum_reciprocal_l3734_373475


namespace existence_of_integers_l3734_373444

theorem existence_of_integers : ∃ (a b c d : ℤ), 
  d ≥ 1 ∧ 
  b % d = c % d ∧ 
  a ∣ b ∧ a ∣ c ∧ 
  (b / a) % d ≠ (c / a) % d := by
  sorry

end existence_of_integers_l3734_373444


namespace max_prob_two_unqualified_expected_cost_min_compensation_fee_l3734_373461

-- Define the probability of a fruit being unqualified
variable (p : ℝ) (hp : 0 < p ∧ p < 1)

-- Define the number of fruits in a box and sample size
def box_size : ℕ := 80
def sample_size : ℕ := 10

-- Define the inspection cost per fruit
def inspection_cost : ℝ := 1.5

-- Define the compensation fee per unqualified fruit
variable (a : ℕ) (ha : a > 0)

-- Function to calculate the probability of exactly k unqualified fruits in a sample of n
def binomial_prob (n k : ℕ) : ℝ → ℝ :=
  λ p => (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

-- Statement 1: Probability that maximizes likelihood of 2 unqualified fruits in 10
theorem max_prob_two_unqualified :
  ∃ p₀, 0 < p₀ ∧ p₀ < 1 ∧
  ∀ p, 0 < p ∧ p < 1 → binomial_prob sample_size 2 p ≤ binomial_prob sample_size 2 p₀ ∧
  p₀ = 0.2 := sorry

-- Statement 2: Expected cost given p = 0.2
theorem expected_cost (p₀ : ℝ) (hp₀ : p₀ = 0.2) :
  (sample_size : ℝ) * inspection_cost + a * (box_size - sample_size : ℝ) * p₀ = 15 + 14 * a := sorry

-- Statement 3: Minimum compensation fee for full inspection
theorem min_compensation_fee :
  ∃ a_min : ℕ, a_min > 0 ∧
  ∀ a : ℕ, a ≥ a_min →
    (box_size : ℝ) * inspection_cost < (sample_size : ℝ) * inspection_cost + a * (box_size - sample_size : ℝ) * 0.2 ∧
  a_min = 8 := sorry

end max_prob_two_unqualified_expected_cost_min_compensation_fee_l3734_373461


namespace unique_base_solution_l3734_373417

/-- Converts a base-10 number to base-a representation --/
def toBaseA (n : ℕ) (a : ℕ) : List ℕ :=
  sorry

/-- Converts a base-a number (represented as a list of digits) to base-10 --/
def fromBaseA (digits : List ℕ) (a : ℕ) : ℕ :=
  sorry

/-- Checks if the equation 452_a + 127_a = 5B0_a holds for a given base a --/
def checkEquation (a : ℕ) : Prop :=
  fromBaseA (toBaseA 452 a) a + fromBaseA (toBaseA 127 a) a = 
  fromBaseA ([5, 11, 0]) a

theorem unique_base_solution :
  ∃! a : ℕ, a > 11 ∧ checkEquation a ∧ fromBaseA ([11]) a = 11 :=
by
  sorry

end unique_base_solution_l3734_373417


namespace science_club_neither_subject_l3734_373481

theorem science_club_neither_subject (total : ℕ) (chemistry : ℕ) (biology : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : chemistry = 42)
  (h3 : biology = 33)
  (h4 : both = 18) :
  total - (chemistry + biology - both) = 18 := by
  sorry

end science_club_neither_subject_l3734_373481


namespace replaced_student_weight_l3734_373488

/-- Given 5 students, if replacing one student with a 72 kg student causes
    the average weight to decrease by 4 kg, then the replaced student's weight was 92 kg. -/
theorem replaced_student_weight
  (n : ℕ) -- number of students
  (old_avg : ℝ) -- original average weight
  (new_avg : ℝ) -- new average weight after replacement
  (new_student_weight : ℝ) -- weight of the new student
  (h1 : n = 5) -- there are 5 students
  (h2 : new_avg = old_avg - 4) -- average weight decreases by 4 kg
  (h3 : new_student_weight = 72) -- new student weighs 72 kg
  : n * old_avg - (n * new_avg + new_student_weight) = 92 := by
  sorry

end replaced_student_weight_l3734_373488


namespace sequence_sum_theorem_l3734_373411

def sequence_term (n : ℕ+) : ℚ := 1 / (n * (n + 1))

def sum_of_terms (n : ℕ+) : ℚ := n / (n + 1)

theorem sequence_sum_theorem (n : ℕ+) :
  (∀ k : ℕ+, k ≤ n → sequence_term k = 1 / (k * (k + 1))) →
  sum_of_terms n = 10 / 11 →
  n = 10 := by sorry

end sequence_sum_theorem_l3734_373411


namespace sqrt_equation_solution_l3734_373403

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 2) = 7 → x = 47 := by
  sorry

end sqrt_equation_solution_l3734_373403


namespace system_of_equations_l3734_373405

theorem system_of_equations (x y a b : ℝ) (h1 : 4*x - 2*y = a) (h2 : 6*y - 12*x = b) (h3 : b ≠ 0) : a/b = -1/3 := by
  sorry

end system_of_equations_l3734_373405


namespace expression_equals_one_l3734_373406

theorem expression_equals_one : 
  (120^2 - 13^2) / (90^2 - 19^2) * ((90-19)*(90+19)) / ((120-13)*(120+13)) = 1 := by
sorry

end expression_equals_one_l3734_373406


namespace rectangular_park_length_l3734_373421

theorem rectangular_park_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 1000) 
  (h2 : breadth = 200) : 
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter ∧ 
  perimeter / 2 - breadth = 300 := by
sorry

end rectangular_park_length_l3734_373421


namespace cricket_average_score_l3734_373498

theorem cricket_average_score 
  (total_matches : ℕ) 
  (overall_average : ℝ) 
  (first_six_average : ℝ) 
  (last_four_sum_lower : ℝ) 
  (last_four_sum_upper : ℝ) 
  (last_four_lowest : ℝ) 
  (h1 : total_matches = 10)
  (h2 : overall_average = 38.9)
  (h3 : first_six_average = 41)
  (h4 : last_four_sum_lower = 120)
  (h5 : last_four_sum_upper = 200)
  (h6 : last_four_lowest = 20)
  (h7 : last_four_sum_lower ≤ (overall_average * total_matches - first_six_average * 6))
  (h8 : (overall_average * total_matches - first_six_average * 6) ≤ last_four_sum_upper) :
  (overall_average * total_matches - first_six_average * 6) / 4 = 35.75 := by
sorry

end cricket_average_score_l3734_373498


namespace distribute_five_balls_four_boxes_l3734_373453

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute 5 4 = 1024 := by
  sorry

end distribute_five_balls_four_boxes_l3734_373453


namespace problem_2017_l3734_373437

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + d * (n - 1)

/-- The problem statement -/
theorem problem_2017 : arithmeticSequence 4 3 672 = 2017 := by
  sorry

end problem_2017_l3734_373437


namespace hexagonal_prism_volume_l3734_373424

-- Define the hexagonal prism
structure HexagonalPrism where
  sideEdgeLength : ℝ
  lateralSurfaceAreaQuadPrism : ℝ

-- Define the theorem
theorem hexagonal_prism_volume 
  (prism : HexagonalPrism)
  (h1 : prism.sideEdgeLength = 3)
  (h2 : prism.lateralSurfaceAreaQuadPrism = 30) :
  ∃ (volume : ℝ), volume = 18 * Real.sqrt 3 := by
  sorry

end hexagonal_prism_volume_l3734_373424


namespace bread_tear_ratio_l3734_373466

/-- Represents the number of bread slices -/
def num_slices : ℕ := 2

/-- Represents the total number of pieces after tearing -/
def total_pieces : ℕ := 8

/-- Represents the number of pieces each slice is torn into -/
def pieces_per_slice : ℕ := total_pieces / num_slices

/-- Proves that the ratio of pieces after the first tear to pieces after the second tear is 1:1 -/
theorem bread_tear_ratio :
  pieces_per_slice = pieces_per_slice → (pieces_per_slice : ℚ) / pieces_per_slice = 1 := by
  sorry

end bread_tear_ratio_l3734_373466


namespace mean_inequality_for_close_numbers_l3734_373443

theorem mean_inequality_for_close_numbers
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxy : x ≠ y)
  (hyz : y ≠ z)
  (hxz : x ≠ z)
  (hclose : ∃ (ε δ : ℝ), ε > 0 ∧ δ > 0 ∧ ε < 1 ∧ δ < 1 ∧ x = y + ε ∧ z = y - δ) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > 2 * y * z / (y + z) :=
sorry

end mean_inequality_for_close_numbers_l3734_373443


namespace hyperbola_properties_l3734_373497

/-- A hyperbola with the given properties -/
def hyperbola (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 = 1

theorem hyperbola_properties :
  ∃ (x y : ℝ),
    -- The hyperbola is centered at the origin
    hyperbola 0 0 ∧
    -- One of its asymptotes is x - 2y = 0
    (∃ (t : ℝ), x = 2*t ∧ y = t) ∧
    -- The hyperbola passes through the point P(√(5/2), 3)
    hyperbola (Real.sqrt (5/2)) 3 :=
by sorry

end hyperbola_properties_l3734_373497


namespace partnership_profit_l3734_373449

theorem partnership_profit (J M : ℕ) (P : ℚ) : 
  J = 700 →
  M = 300 →
  (P / 6 + (J * 2 * P) / (3 * (J + M))) - (P / 6 + (M * 2 * P) / (3 * (J + M))) = 800 →
  P = 3000 := by
sorry

end partnership_profit_l3734_373449


namespace cashier_money_value_l3734_373412

def total_bills : ℕ := 30
def ten_dollar_bills : ℕ := 27
def twenty_dollar_bills : ℕ := 3
def ten_dollar_value : ℕ := 10
def twenty_dollar_value : ℕ := 20

theorem cashier_money_value :
  ten_dollar_bills + twenty_dollar_bills = total_bills →
  ten_dollar_bills * ten_dollar_value + twenty_dollar_bills * twenty_dollar_value = 330 :=
by
  sorry

end cashier_money_value_l3734_373412


namespace cubic_inequality_solution_l3734_373470

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 10*x^2 + 28*x > 0 ↔ (x > 0 ∧ x < 4) ∨ x > 6 := by
  sorry

end cubic_inequality_solution_l3734_373470


namespace population_increase_rate_l3734_373436

theorem population_increase_rate 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (increase_rate : ℚ) :
  initial_population = 2000 →
  final_population = 2400 →
  increase_rate = (final_population - initial_population) / initial_population * 100 →
  increase_rate = 20 := by
  sorry

end population_increase_rate_l3734_373436


namespace weight_distribution_l3734_373468

theorem weight_distribution :
  ∃! (x y z : ℕ), x + y + z = 11 ∧ 3 * x + 7 * y + 14 * z = 108 :=
by
  sorry

end weight_distribution_l3734_373468


namespace common_root_not_implies_equal_coefficients_l3734_373459

theorem common_root_not_implies_equal_coefficients
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + b * x + c = 0 ∧ c * x^2 + b * x + a = 0) → ¬(a = c) :=
sorry

end common_root_not_implies_equal_coefficients_l3734_373459


namespace mean_of_five_numbers_l3734_373429

theorem mean_of_five_numbers (a b c d e : ℚ) 
  (sum_condition : a + b + c + d + e = 3/4) :
  (a + b + c + d + e) / 5 = 3/20 := by
  sorry

end mean_of_five_numbers_l3734_373429


namespace cartesian_angle_properties_l3734_373467

/-- An angle in the Cartesian coordinate system -/
structure CartesianAngle where
  /-- The x-coordinate of the point on the terminal side -/
  x : ℝ
  /-- The y-coordinate of the point on the terminal side -/
  y : ℝ

/-- Theorem about properties of a specific angle in the Cartesian coordinate system -/
theorem cartesian_angle_properties (α : CartesianAngle) 
  (h1 : α.x = -1) 
  (h2 : α.y = 2) : 
  (Real.sin α.y * Real.tan α.y = -4 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (α.y + Real.pi / 2) * Real.cos (7 * Real.pi / 2 - α.y) * Real.tan (2 * Real.pi - α.y)) / 
   (Real.sin (2 * Real.pi - α.y) * Real.tan (-α.y)) = -Real.sqrt 5 / 5) :=
by sorry

end cartesian_angle_properties_l3734_373467


namespace initial_short_trees_count_l3734_373458

/-- The number of short trees in the park after planting -/
def final_short_trees : ℕ := 95

/-- The number of short trees planted today -/
def planted_short_trees : ℕ := 64

/-- The initial number of short trees in the park -/
def initial_short_trees : ℕ := final_short_trees - planted_short_trees

theorem initial_short_trees_count : initial_short_trees = 31 := by
  sorry

end initial_short_trees_count_l3734_373458


namespace tree_height_after_four_years_l3734_373477

/-- The height of a tree that doubles every year -/
def treeHeight (initialHeight : ℝ) (years : ℕ) : ℝ :=
  initialHeight * (2 ^ years)

theorem tree_height_after_four_years
  (h : treeHeight 1 7 = 64) :
  treeHeight 1 4 = 8 :=
sorry

end tree_height_after_four_years_l3734_373477


namespace lily_petals_l3734_373450

theorem lily_petals (num_lilies : ℕ) (num_tulips : ℕ) (tulip_petals : ℕ) (total_petals : ℕ) :
  num_lilies = 8 →
  num_tulips = 5 →
  tulip_petals = 3 →
  total_petals = 63 →
  ∃ (lily_petals : ℕ), lily_petals * num_lilies + tulip_petals * num_tulips = total_petals ∧ lily_petals = 6 :=
by sorry

end lily_petals_l3734_373450


namespace total_students_l3734_373423

theorem total_students (girls : ℕ) (boys : ℕ) (total : ℕ) : 
  girls = 160 →
  5 * boys = 8 * girls →
  total = girls + boys →
  total = 416 := by
sorry

end total_students_l3734_373423


namespace polynomial_roots_arithmetic_progression_l3734_373483

theorem polynomial_roots_arithmetic_progression 
  (a b : ℝ) 
  (ha : a = 3 * Real.sqrt 3) 
  (hroots : ∀ (r s t : ℝ), 
    (r^3 - a*r^2 + b*r + a = 0 ∧ 
     s^3 - a*s^2 + b*s + a = 0 ∧ 
     t^3 - a*t^2 + b*t + a = 0) → 
    (r > 0 ∧ s > 0 ∧ t > 0) ∧ 
    ∃ (d : ℝ), (s = r + d ∧ t = r + 2*d) ∨ (s = r ∧ t = r)) : 
  b = 3 * (Real.sqrt 3 + 1) := by
sorry

end polynomial_roots_arithmetic_progression_l3734_373483


namespace madeline_work_hours_l3734_373448

def rent : ℕ := 1200
def groceries : ℕ := 400
def medical : ℕ := 200
def utilities : ℕ := 60
def emergency : ℕ := 200
def hourly_wage : ℕ := 15

def total_expenses : ℕ := rent + groceries + medical + utilities + emergency

def hours_needed : ℕ := (total_expenses + hourly_wage - 1) / hourly_wage

theorem madeline_work_hours :
  hours_needed = 138 :=
sorry

end madeline_work_hours_l3734_373448


namespace polynomial_coefficient_sum_l3734_373428

theorem polynomial_coefficient_sum (a b c d : ℤ) : 
  (∀ x : ℚ, (3*x + 2) * (2*x - 3) * (x - 4) = a*x^3 + b*x^2 + c*x + d) →
  a - b + c - d = 25 := by
sorry

end polynomial_coefficient_sum_l3734_373428


namespace multiplication_division_equality_l3734_373499

theorem multiplication_division_equality : (3 * 4) / 6 = 2 := by
  sorry

end multiplication_division_equality_l3734_373499


namespace divisibility_of_sum_and_powers_l3734_373465

theorem divisibility_of_sum_and_powers (a b c : ℤ) : 
  (6 ∣ (a + b + c)) → (6 ∣ (a^5 + b^3 + c)) := by sorry

end divisibility_of_sum_and_powers_l3734_373465


namespace logarithm_identity_l3734_373473

theorem logarithm_identity (a : ℝ) (ha : a > 0) : 
  a^(Real.log (Real.log a)) - (Real.log a)^(Real.log a) = 0 :=
by sorry

end logarithm_identity_l3734_373473


namespace sin_product_equals_sqrt5_minus_1_over_32_sin_cos_ratio_equals_neg_sqrt2_l3734_373451

-- Part 1
theorem sin_product_equals_sqrt5_minus_1_over_32 :
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 
  (Real.sqrt 5 - 1) / 32 := by sorry

-- Part 2
theorem sin_cos_ratio_equals_neg_sqrt2 (α : Real) 
  (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + π / 4)) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = -Real.sqrt 2 := by sorry

end sin_product_equals_sqrt5_minus_1_over_32_sin_cos_ratio_equals_neg_sqrt2_l3734_373451


namespace expansion_coefficients_l3734_373447

theorem expansion_coefficients (m : ℝ) (n : ℕ) :
  m > 0 →
  (1 : ℝ) + n + (n * (n - 1) / 2) = 37 →
  m ^ 2 * (Nat.choose n 6) = 112 →
  n = 8 ∧ m = 2 := by sorry

end expansion_coefficients_l3734_373447


namespace functional_equation_bijection_l3734_373427

theorem functional_equation_bijection :
  ∃ f : ℕ → ℕ, Function.Bijective f ∧
    ∀ m n : ℕ, f (3*m*n + m + n) = 4*f m*f n + f m + f n :=
by sorry

end functional_equation_bijection_l3734_373427


namespace series_sum_equals_four_ninths_l3734_373464

theorem series_sum_equals_four_ninths :
  (∑' n : ℕ, n / (4 : ℝ)^n) = 4 / 9 := by
  sorry

end series_sum_equals_four_ninths_l3734_373464


namespace triangle_inequality_equivalence_l3734_373441

theorem triangle_inequality_equivalence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4) :=
sorry

end triangle_inequality_equivalence_l3734_373441


namespace prime_sum_equality_l3734_373492

theorem prime_sum_equality (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p * (p + 1) + q * (q + 1) = n * (n + 1)) → 
  n = 3 ∨ n = 6 := by
sorry

end prime_sum_equality_l3734_373492


namespace total_arrangements_eq_65_l3734_373486

-- Define the number of seats
def num_seats : ℕ := 10

-- Define the number of women
def num_women : ℕ := 6

-- Define the number of men
def num_men : ℕ := 1

-- Define the maximum number of seats a woman can move
def max_women_move : ℕ := 1

-- Define the maximum number of seats a man can move
def max_men_move : ℕ := 2

-- Define a function to calculate the number of reseating arrangements for women
def women_arrangements (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => women_arrangements (n + 1) + women_arrangements n

-- Define a function to calculate the number of reseating arrangements for the man
def man_arrangements (original_pos : ℕ) : ℕ :=
  2 * max_men_move + 1

-- Theorem: The total number of reseating arrangements is 65
theorem total_arrangements_eq_65 :
  women_arrangements num_women * man_arrangements (num_women + 1) = 65 := by
  sorry

end total_arrangements_eq_65_l3734_373486


namespace base_b_problem_l3734_373484

theorem base_b_problem (b : ℕ) : 
  (6 * b^2 + 5 * b + 5 = (2 * b + 5)^2) → b = 9 := by
  sorry

end base_b_problem_l3734_373484
