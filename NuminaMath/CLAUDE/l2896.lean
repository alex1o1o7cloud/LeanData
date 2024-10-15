import Mathlib

namespace NUMINAMATH_CALUDE_expand_expression_l2896_289658

theorem expand_expression (x y : ℝ) : (x + 10) * (2 * y + 10) = 2 * x * y + 10 * x + 20 * y + 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2896_289658


namespace NUMINAMATH_CALUDE_chetan_score_percentage_l2896_289694

theorem chetan_score_percentage (max_score : ℕ) (amar_percent : ℚ) (bhavan_percent : ℚ) (average_mark : ℕ) :
  max_score = 900 →
  amar_percent = 64/100 →
  bhavan_percent = 36/100 →
  average_mark = 432 →
  ∃ (chetan_percent : ℚ), 
    (amar_percent + bhavan_percent + chetan_percent) * max_score / 3 = average_mark ∧
    chetan_percent = 44/100 :=
by sorry

end NUMINAMATH_CALUDE_chetan_score_percentage_l2896_289694


namespace NUMINAMATH_CALUDE_no_prime_5n_plus_3_l2896_289669

theorem no_prime_5n_plus_3 : ¬∃ (n : ℕ+), 
  (∃ k : ℕ, (2 : ℤ) * n + 1 = k^2) ∧ 
  (∃ l : ℕ, (3 : ℤ) * n + 1 = l^2) ∧ 
  Nat.Prime ((5 : ℤ) * n + 3).toNat :=
by sorry

end NUMINAMATH_CALUDE_no_prime_5n_plus_3_l2896_289669


namespace NUMINAMATH_CALUDE_equation_solution_l2896_289621

theorem equation_solution (x : ℝ) : 3 / (x + 10) = 1 / (2 * x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2896_289621


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l2896_289641

def hulk_jump (n : ℕ) : ℝ := 3^n

theorem hulk_jump_exceeds_1000 : 
  (∀ k < 7, hulk_jump k ≤ 1000) ∧ hulk_jump 7 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l2896_289641


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2896_289687

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 66 ways to put 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : ways_to_distribute 5 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2896_289687


namespace NUMINAMATH_CALUDE_flute_cost_l2896_289615

/-- Calculates the cost of a flute given the total amount spent and the costs of other items --/
theorem flute_cost (total_spent music_stand_cost song_book_cost : ℚ) : 
  total_spent = 158.35 ∧ music_stand_cost = 8.89 ∧ song_book_cost = 7 →
  total_spent - (music_stand_cost + song_book_cost) = 142.46 := by
  sorry

end NUMINAMATH_CALUDE_flute_cost_l2896_289615


namespace NUMINAMATH_CALUDE_divisibility_implication_l2896_289681

theorem divisibility_implication (x y : ℤ) : 
  (23 ∣ (3 * x + 2 * y)) → (23 ∣ (17 * x + 19 * y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2896_289681


namespace NUMINAMATH_CALUDE_no_single_common_tangent_for_equal_circles_l2896_289604

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a function to count common tangents between two circles
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem no_single_common_tangent_for_equal_circles (c1 c2 : Circle) :
  c1.radius = c2.radius → c1 ≠ c2 → countCommonTangents c1 c2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_single_common_tangent_for_equal_circles_l2896_289604


namespace NUMINAMATH_CALUDE_alternating_sum_equals_eight_l2896_289638

theorem alternating_sum_equals_eight :
  43 - 41 + 39 - 37 + 35 - 33 + 31 - 29 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_equals_eight_l2896_289638


namespace NUMINAMATH_CALUDE_oranges_distribution_l2896_289654

theorem oranges_distribution (total : ℕ) (boxes : ℕ) (difference : ℕ) (first_box : ℕ) : 
  total = 120 →
  boxes = 7 →
  difference = 2 →
  (first_box * boxes + (boxes * (boxes - 1) * difference) / 2 = total) →
  first_box = 11 := by
sorry

end NUMINAMATH_CALUDE_oranges_distribution_l2896_289654


namespace NUMINAMATH_CALUDE_race_outcomes_l2896_289677

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_positions : ℕ := 3

/-- The number of different podium outcomes in a race with no ties -/
def num_outcomes : ℕ := num_participants * (num_participants - 1) * (num_participants - 2)

theorem race_outcomes :
  num_outcomes = 120 :=
by sorry

end NUMINAMATH_CALUDE_race_outcomes_l2896_289677


namespace NUMINAMATH_CALUDE_activity_popularity_ranking_l2896_289606

/-- Represents the popularity of an activity as a fraction --/
structure ActivityPopularity where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- The three activities in the festival --/
inductive Activity
  | Dance
  | Painting
  | ClayModeling

/-- Given popularity data for the activities --/
def popularity : Activity → ActivityPopularity
  | Activity.Dance => ⟨3, 8, by norm_num⟩
  | Activity.Painting => ⟨5, 16, by norm_num⟩
  | Activity.ClayModeling => ⟨9, 24, by norm_num⟩

/-- Convert a fraction to a common denominator --/
def toCommonDenominator (ap : ActivityPopularity) (lcd : ℕ) : ℚ :=
  (ap.numerator : ℚ) * (lcd / ap.denominator) / lcd

/-- The least common denominator of all activities' fractions --/
def leastCommonDenominator : ℕ := 48

theorem activity_popularity_ranking :
  let commonDance := toCommonDenominator (popularity Activity.Dance) leastCommonDenominator
  let commonPainting := toCommonDenominator (popularity Activity.Painting) leastCommonDenominator
  let commonClayModeling := toCommonDenominator (popularity Activity.ClayModeling) leastCommonDenominator
  (commonDance = commonClayModeling) ∧ (commonDance > commonPainting) := by
  sorry

#check activity_popularity_ranking

end NUMINAMATH_CALUDE_activity_popularity_ranking_l2896_289606


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_focus_coordinates_y_eq_2x_squared_l2896_289662

/-- The focus of a parabola y = ax^2 has coordinates (0, 1/(4a)) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - a * x^2
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1 / (4 * a) ∧ 
    ∀ (x y : ℝ), f (x, y) = 0 → (x - p.1)^2 + (y - p.2)^2 = (y - p.2 + 1 / (4 * a))^2 :=
sorry

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem focus_coordinates_y_eq_2x_squared :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - 2 * x^2
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1/8 ∧ 
    ∀ (x y : ℝ), f (x, y) = 0 → (x - p.1)^2 + (y - p.2)^2 = (y - p.2 + 1/8)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_focus_coordinates_y_eq_2x_squared_l2896_289662


namespace NUMINAMATH_CALUDE_range_of_a_l2896_289639

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 > 2*x - 1) : a ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2896_289639


namespace NUMINAMATH_CALUDE_factor_implies_h_value_l2896_289686

theorem factor_implies_h_value (m h : ℝ) : 
  (∃ k : ℝ, m^2 - h*m - 24 = (m - 8) * k) → h = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_h_value_l2896_289686


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l2896_289660

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l2896_289660


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2896_289697

theorem positive_real_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + 3*b^2 ≥ 2*b*(a + b)) ∧ (a^3 + b^3 ≥ a*b^2 + a^2*b) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2896_289697


namespace NUMINAMATH_CALUDE_unique_modular_solution_l2896_289627

theorem unique_modular_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -567 [ZMOD 13] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l2896_289627


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2896_289698

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2896_289698


namespace NUMINAMATH_CALUDE_goldfish_remaining_l2896_289611

/-- Given Finn's initial number of goldfish and the number that die,
    prove the number of goldfish left. -/
theorem goldfish_remaining (initial : ℕ) (died : ℕ) :
  initial ≥ died →
  initial - died = initial - died :=
by sorry

end NUMINAMATH_CALUDE_goldfish_remaining_l2896_289611


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2896_289692

theorem fraction_subtraction (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x^2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x^2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x^2 + 3 * x - 5) / ((x + 2) * (x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2896_289692


namespace NUMINAMATH_CALUDE_value_of_y_l2896_289676

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2896_289676


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2896_289640

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ x : ℝ, (1 + x > a ∧ 2 * x - 4 ≤ 0)) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2896_289640


namespace NUMINAMATH_CALUDE_kaleb_shirts_l2896_289602

-- Define the initial number of shirts
def initial_shirts : ℕ := 17

-- Define the number of shirts Kaleb would have after getting rid of 7
def remaining_shirts : ℕ := 10

-- Define the number of shirts Kaleb got rid of
def removed_shirts : ℕ := 7

-- Theorem to prove
theorem kaleb_shirts : initial_shirts = remaining_shirts + removed_shirts :=
by sorry

end NUMINAMATH_CALUDE_kaleb_shirts_l2896_289602


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l2896_289608

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l2896_289608


namespace NUMINAMATH_CALUDE_hash_difference_l2896_289656

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 8 5 - hash 5 8 = -12 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l2896_289656


namespace NUMINAMATH_CALUDE_unknown_number_in_set_l2896_289685

theorem unknown_number_in_set (x : ℝ) : 
  let set1 : List ℝ := [12, 32, 56, 78, 91]
  let set2 : List ℝ := [7, 47, 67, 105, x]
  (set1.sum / set1.length : ℝ) = (set2.sum / set2.length : ℝ) + 10 →
  x = -7 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_in_set_l2896_289685


namespace NUMINAMATH_CALUDE_digits_of_8_pow_20_times_5_pow_18_l2896_289671

/-- The number of digits in a positive integer n in base b -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

/-- Theorem: The number of digits in 8^20 * 5^18 in base 10 is 31 -/
theorem digits_of_8_pow_20_times_5_pow_18 :
  num_digits (8^20 * 5^18) 10 = 31 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_8_pow_20_times_5_pow_18_l2896_289671


namespace NUMINAMATH_CALUDE_sum_of_fractions_nonnegative_l2896_289680

theorem sum_of_fractions_nonnegative (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + 
  (33 * b^2 - b) / (33 * b^2 + 1) + 
  (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_nonnegative_l2896_289680


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_one_l2896_289630

theorem purely_imaginary_iff_a_eq_one (a : ℝ) : 
  (Complex.I * (a * Complex.I) = (a^2 - a) + a * Complex.I) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_one_l2896_289630


namespace NUMINAMATH_CALUDE_max_distance_complex_l2896_289678

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z - 1) = 1) :
  ∃ (max_val : ℝ), max_val = 3 ∧ ∀ w, Complex.abs (w - 1) = 1 → Complex.abs (w - (2 * Complex.I + 1)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2896_289678


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2896_289646

theorem unique_solution_condition (a b : ℤ) : 
  (∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b) ↔ 4 * b = a^2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2896_289646


namespace NUMINAMATH_CALUDE_congruence_square_implies_congruence_or_negative_l2896_289679

theorem congruence_square_implies_congruence_or_negative (x y : ℤ) :
  x^2 ≡ y^2 [ZMOD 239] → (x ≡ y [ZMOD 239] ∨ x ≡ -y [ZMOD 239]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_square_implies_congruence_or_negative_l2896_289679


namespace NUMINAMATH_CALUDE_john_business_venture_result_l2896_289691

structure Currency where
  name : String
  exchange_rate : ℚ

structure Item where
  name : String
  currency : Currency
  purchase_price : ℚ
  sale_percentage : ℚ
  tax_rate : ℚ

def calculate_profit_or_loss (items : List Item) : ℚ :=
  sorry

theorem john_business_venture_result 
  (grinder : Item)
  (mobile_phone : Item)
  (refrigerator : Item)
  (television : Item)
  (h_grinder : grinder = { 
    name := "Grinder", 
    currency := { name := "INR", exchange_rate := 1 },
    purchase_price := 15000,
    sale_percentage := -4/100,
    tax_rate := 5/100
  })
  (h_mobile_phone : mobile_phone = {
    name := "Mobile Phone",
    currency := { name := "USD", exchange_rate := 75 },
    purchase_price := 100,
    sale_percentage := 10/100,
    tax_rate := 7/100
  })
  (h_refrigerator : refrigerator = {
    name := "Refrigerator",
    currency := { name := "GBP", exchange_rate := 101 },
    purchase_price := 200,
    sale_percentage := 8/100,
    tax_rate := 6/100
  })
  (h_television : television = {
    name := "Television",
    currency := { name := "EUR", exchange_rate := 90 },
    purchase_price := 300,
    sale_percentage := -6/100,
    tax_rate := 9/100
  }) :
  calculate_profit_or_loss [grinder, mobile_phone, refrigerator, television] = -346/100 :=
sorry

end NUMINAMATH_CALUDE_john_business_venture_result_l2896_289691


namespace NUMINAMATH_CALUDE_total_payment_l2896_289659

def payment_structure (year1 year2 year3 year4 : ℕ) : Prop :=
  year1 = 20 ∧ 
  year2 = year1 + 2 ∧ 
  year3 = year2 + 3 ∧ 
  year4 = year3 + 4

theorem total_payment (year1 year2 year3 year4 : ℕ) :
  payment_structure year1 year2 year3 year4 →
  year1 + year2 + year3 + year4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_l2896_289659


namespace NUMINAMATH_CALUDE_function_value_problem_l2896_289614

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f (x / 2 - 1) = 2 * x + 3) →
  f m = 6 →
  m = -1/4 := by
    sorry

end NUMINAMATH_CALUDE_function_value_problem_l2896_289614


namespace NUMINAMATH_CALUDE_line_parametric_equations_l2896_289665

/-- Parametric equations of a line passing through M(1,5) with inclination angle 2π/3 -/
theorem line_parametric_equations (t : ℝ) : 
  let M : ℝ × ℝ := (1, 5)
  let angle : ℝ := 2 * Real.pi / 3
  let P : ℝ × ℝ := (1 - (1/2) * t, 5 + (Real.sqrt 3 / 2) * t)
  (P.1 - M.1 = t * Real.cos angle) ∧ (P.2 - M.2 = t * Real.sin angle) := by
  sorry

end NUMINAMATH_CALUDE_line_parametric_equations_l2896_289665


namespace NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2896_289644

/-- Given a right circular cylinder with radius 2 and a plane intersecting it to form an ellipse,
    if the major axis of the ellipse is 25% longer than its minor axis,
    then the length of the major axis is 5. -/
theorem cylinder_ellipse_intersection (cylinder_radius : ℝ) (minor_axis major_axis : ℝ) :
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = 1.25 * minor_axis →
  major_axis = 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2896_289644


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2896_289601

theorem min_value_squared_sum (a b c : ℝ) (h : 2*a + 2*b + c = 8) :
  (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ≥ 49/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2896_289601


namespace NUMINAMATH_CALUDE_distance_in_scientific_notation_l2896_289652

/-- Given a distance of 14,000,000 meters between two mountain peaks,
    prove that its representation in scientific notation is 1.4 × 10^7 -/
theorem distance_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    14000000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧ 
    a = 1.4 ∧ 
    n = 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_in_scientific_notation_l2896_289652


namespace NUMINAMATH_CALUDE_truck_distance_l2896_289634

theorem truck_distance (truck_time car_time : ℝ) (speed_difference : ℝ) :
  truck_time = 8 →
  car_time = 5 →
  speed_difference = 18 →
  ∃ (truck_speed : ℝ),
    truck_speed * truck_time = (truck_speed + speed_difference) * car_time ∧
    truck_speed * truck_time = 240 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l2896_289634


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2896_289689

/-- Given a polar equation ρ = 4sin(θ), prove its equivalence to the Cartesian equation x² + (y-2)² = 4 -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2896_289689


namespace NUMINAMATH_CALUDE_base_7_to_decimal_l2896_289628

/-- Converts a list of digits in base b to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The base 7 number 3206 -/
def base_7_number : List Nat := [3, 2, 0, 6]

/-- Theorem stating that 3206 in base 7 is equal to 1133 in base 10 -/
theorem base_7_to_decimal :
  to_decimal base_7_number 7 = 1133 := by
  sorry

end NUMINAMATH_CALUDE_base_7_to_decimal_l2896_289628


namespace NUMINAMATH_CALUDE_complex_dot_product_l2896_289635

theorem complex_dot_product (z : ℂ) (h1 : Complex.abs z = Real.sqrt 2) (h2 : Complex.im (z^2) = 2) :
  (z + z^2) • (z - z^2) = -2 ∨ (z + z^2) • (z - z^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_dot_product_l2896_289635


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2896_289600

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2896_289600


namespace NUMINAMATH_CALUDE_jellybean_count_l2896_289699

theorem jellybean_count (nephews nieces jellybeans_per_child : ℕ) 
  (h1 : nephews = 3)
  (h2 : nieces = 2)
  (h3 : jellybeans_per_child = 14) :
  (nephews + nieces) * jellybeans_per_child = 70 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l2896_289699


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l2896_289612

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : Real) : Real :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : Real := 4
  let boat_breadth : Real := 3
  let boat_sink_height : Real := 0.01
  let water_density : Real := 1000
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 120 := by
  sorry

#check mass_of_man_on_boat

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l2896_289612


namespace NUMINAMATH_CALUDE_perpendicular_m_value_parallel_distance_l2896_289647

-- Define the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (m - 3) * y + m = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x - 2 * y + 4 = 0

-- Define perpendicularity condition
def perpendicular (m : ℝ) : Prop := (-1 : ℝ) / (m - 3) * (m / 2) = -1

-- Define parallelism condition
def parallel (m : ℝ) : Prop := 1 * (-2) = m * (m - 3)

-- Theorem for perpendicular case
theorem perpendicular_m_value : ∃ m : ℝ, perpendicular m ∧ m = 6 := by sorry

-- Theorem for parallel case
theorem parallel_distance : 
  ∃ m : ℝ, parallel m ∧ 
  (let d := |4 - 1| / Real.sqrt (1^2 + (-2)^2);
   d = 3 * Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_perpendicular_m_value_parallel_distance_l2896_289647


namespace NUMINAMATH_CALUDE_quadratic_complete_square_sum_l2896_289642

/-- Given a quadratic equation x^2 - 2x + m = 0 that can be written as (x-1)^2 = n
    after completing the square, prove that m + n = 1 -/
theorem quadratic_complete_square_sum (m n : ℝ) : 
  (∀ x, x^2 - 2*x + m = 0 ↔ (x - 1)^2 = n) → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_sum_l2896_289642


namespace NUMINAMATH_CALUDE_cube_coverage_l2896_289613

/-- Represents a paper strip of size 3 × 1 -/
structure PaperStrip :=
  (length : Nat := 3)
  (width : Nat := 1)

/-- Represents a cube of size n × n × n -/
structure Cube (n : Nat) :=
  (side_length : Nat := n)

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : Nat) : Prop := n % 3 = 0

/-- Predicate to check if it's possible to cover three sides of a cube with paper strips -/
def can_cover_sides (c : Cube n) (p : PaperStrip) : Prop :=
  divisible_by_three n

/-- Theorem stating the condition for covering three sides of a cube with paper strips -/
theorem cube_coverage (n : Nat) :
  ∀ (c : Cube n) (p : PaperStrip),
    can_cover_sides c p ↔ divisible_by_three n :=
by sorry

end NUMINAMATH_CALUDE_cube_coverage_l2896_289613


namespace NUMINAMATH_CALUDE_binomial_16_choose_5_l2896_289688

theorem binomial_16_choose_5 : Nat.choose 16 5 = 4368 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_choose_5_l2896_289688


namespace NUMINAMATH_CALUDE_time_to_hospital_l2896_289651

/-- Proves that given a distance of 0.09 kilometers to the hospital and a speed of 3 meters per 4 seconds, it takes 120 seconds for Ayeon to reach the hospital. -/
theorem time_to_hospital (distance_km : ℝ) (speed_m : ℝ) (speed_s : ℝ) : 
  distance_km = 0.09 →
  speed_m = 3 →
  speed_s = 4 →
  (distance_km * 1000) / (speed_m / speed_s) = 120 := by
sorry

end NUMINAMATH_CALUDE_time_to_hospital_l2896_289651


namespace NUMINAMATH_CALUDE_geometric_relations_l2896_289624

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelLL : Line → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularLL : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem geometric_relations 
  (a b c : Line) (α β γ : Plane) :
  -- Proposition 2
  (skew a b ∧ 
   contains α a ∧ 
   contains β b ∧ 
   parallelLP a β ∧ 
   parallelLP b α → 
   parallel α β) ∧
  -- Proposition 3
  (intersect α β a ∧ 
   intersect β γ b ∧ 
   intersect γ α c ∧ 
   parallelLL a b → 
   parallelLP c β) ∧
  -- Proposition 4
  (skew a b ∧ 
   parallelLP a α ∧ 
   parallelLP b α ∧ 
   perpendicularLL c a ∧ 
   perpendicularLL c b → 
   perpendicularLP c α) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l2896_289624


namespace NUMINAMATH_CALUDE_log_domain_intersection_l2896_289632

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem log_domain_intersection :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_log_domain_intersection_l2896_289632


namespace NUMINAMATH_CALUDE_amy_garden_problem_l2896_289666

/-- Amy's gardening problem -/
theorem amy_garden_problem (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)
  (h1 : total_seeds = 101)
  (h2 : small_gardens = 9)
  (h3 : seeds_per_small_garden = 6) :
  total_seeds - (small_gardens * seeds_per_small_garden) = 47 := by
  sorry

end NUMINAMATH_CALUDE_amy_garden_problem_l2896_289666


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2896_289607

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 74 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 17.6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2896_289607


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2896_289617

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = 0.56) ∧ x = 56/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2896_289617


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l2896_289610

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'O', 'X', 'O', 'X', 'O']

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 
  (1 : ℚ) / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l2896_289610


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l2896_289626

theorem power_of_eight_sum_equals_power_of_two : 8^17 + 8^17 + 8^17 + 8^17 = 2^53 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l2896_289626


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l2896_289696

/-- The minimum distance from a point on the circle x^2 + y^2 = 1 to the line 3x + 4y - 25 = 0 is 4 -/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y - 25 = 0}
  ∃ (d : ℝ), d = 4 ∧ ∀ (p : ℝ × ℝ), p ∈ circle →
    ∀ (q : ℝ × ℝ), q ∈ line →
      d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l2896_289696


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l2896_289623

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem cubic_function_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (y : ℝ), f a y ≤ f a x₁) ∧ 
    (∀ (y : ℝ), f a y ≥ f a x₂)) →
  (a < -3 ∨ a > 6) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l2896_289623


namespace NUMINAMATH_CALUDE_soris_population_2080_l2896_289663

/-- The population growth function for Soris island -/
def soris_population (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (2 ^ (years_passed / 20))

/-- Theorem stating the population of Soris in 2080 -/
theorem soris_population_2080 :
  soris_population 500 80 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_soris_population_2080_l2896_289663


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2896_289653

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 5) / (x - 3) = (x - 4) / (x + 2) ↔ x = 1 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2896_289653


namespace NUMINAMATH_CALUDE_second_project_questions_l2896_289609

/-- Calculates the number of questions for the second project given the total questions per day,
    number of days, and questions for the first project. -/
def questions_for_second_project (questions_per_day : ℕ) (days : ℕ) (questions_first_project : ℕ) : ℕ :=
  questions_per_day * days - questions_first_project

/-- Proves that given the specified conditions, the number of questions for the second project is 476. -/
theorem second_project_questions :
  questions_for_second_project 142 7 518 = 476 := by
  sorry

end NUMINAMATH_CALUDE_second_project_questions_l2896_289609


namespace NUMINAMATH_CALUDE_arithmetic_geometric_properties_l2896_289682

/-- Given an arithmetic progression {a_n} with common difference d,
    where a_3, a_4, and a_8 form a geometric progression,
    prove certain properties about the sequence and its sum. -/
theorem arithmetic_geometric_properties
  (a : ℕ → ℝ)  -- The sequence a_n
  (d : ℝ)      -- Common difference
  (h1 : d ≠ 0) -- d is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d)  -- Arithmetic progression property
  (h3 : (a 4) ^ 2 = a 3 * a 8)     -- Geometric progression property
  : a 1 * d < 0 ∧ 
    d * (4 * a 1 + 6 * d) < 0 ∧
    (a 4 / a 3 = 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_properties_l2896_289682


namespace NUMINAMATH_CALUDE_correct_answer_calculation_l2896_289695

theorem correct_answer_calculation (x y : ℝ) : 
  (y = x + 2 * 0.42) → (x = y - 2 * 0.42) :=
by
  sorry

#eval (0.9 : ℝ) - 2 * 0.42

end NUMINAMATH_CALUDE_correct_answer_calculation_l2896_289695


namespace NUMINAMATH_CALUDE_prime_sum_2003_l2896_289648

theorem prime_sum_2003 (a b : ℕ) (ha : Prime a) (hb : Prime b) (h : a^2 + b = 2003) : 
  a + b = 2001 := by sorry

end NUMINAMATH_CALUDE_prime_sum_2003_l2896_289648


namespace NUMINAMATH_CALUDE_fraction_comparison_l2896_289668

theorem fraction_comparison : ((3 / 5 : ℚ) * 320 + (5 / 9 : ℚ) * 540) - ((7 / 12 : ℚ) * 450) = 229.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2896_289668


namespace NUMINAMATH_CALUDE_eighth_row_interior_sum_l2896_289693

/-- Sum of all elements in the n-th row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem eighth_row_interior_sum :
  pascal_interior_sum 8 = 126 := by sorry

end NUMINAMATH_CALUDE_eighth_row_interior_sum_l2896_289693


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l2896_289633

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 27, prove that the difference 
between the two digits of the number is 3.
-/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 27 → x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l2896_289633


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2896_289605

theorem tangent_line_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x → (∃! x₀ y₀ : ℝ, y₀ = 3*x₀ + c ∧ y₀^2 = 12*x₀)) → 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2896_289605


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2896_289655

theorem sufficient_not_necessary :
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) ∧
  (∀ x : ℝ, x > 2 → x^2 > 4) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2896_289655


namespace NUMINAMATH_CALUDE_exists_term_with_nine_l2896_289603

/-- An arithmetic progression with natural number first term and common difference -/
structure ArithmeticProgression :=
  (first_term : ℕ)
  (common_difference : ℕ)

/-- Function to check if a natural number contains the digit 9 -/
def contains_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 9

/-- Theorem stating that there exists a term in the arithmetic progression containing the digit 9 -/
theorem exists_term_with_nine (ap : ArithmeticProgression) :
  ∃ n : ℕ, contains_nine (ap.first_term + n * ap.common_difference) :=
sorry

end NUMINAMATH_CALUDE_exists_term_with_nine_l2896_289603


namespace NUMINAMATH_CALUDE_triangle_triplets_characterization_l2896_289657

def is_valid_triplet (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧
  (∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r) ∧
  (a = 100 ∨ c = 100)

def valid_triplets : Set (ℕ × ℕ × ℕ) :=
  {(49,70,100), (64,80,100), (81,90,100), (100,100,100), (100,110,121),
   (100,120,144), (100,130,169), (100,140,196), (100,150,225), (100,160,256)}

theorem triangle_triplets_characterization :
  {(a, b, c) | is_valid_triplet a b c} = valid_triplets :=
by sorry

end NUMINAMATH_CALUDE_triangle_triplets_characterization_l2896_289657


namespace NUMINAMATH_CALUDE_harper_consumption_l2896_289650

/-- Represents the mineral water consumption problem -/
structure MineralWaterConsumption where
  bottles_per_case : ℕ
  cost_per_case : ℚ
  total_spent : ℚ
  days_supply : ℕ

/-- Calculates the daily mineral water consumption given the problem parameters -/
def daily_consumption (m : MineralWaterConsumption) : ℚ :=
  (m.total_spent / m.cost_per_case * m.bottles_per_case) / m.days_supply

/-- Theorem stating that Harper's daily mineral water consumption is 0.5 bottles -/
theorem harper_consumption :
  ∃ (m : MineralWaterConsumption),
    m.bottles_per_case = 24 ∧
    m.cost_per_case = 12 ∧
    m.total_spent = 60 ∧
    m.days_supply = 240 ∧
    daily_consumption m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_harper_consumption_l2896_289650


namespace NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_minus_1_l2896_289631

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 1 else Real.log x / Real.log 3 + 1

theorem f_composition_equals_sqrt2_over_2_minus_1 :
  f (f (Real.sqrt 3 / 9)) = Real.sqrt 2 / 2 - 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_minus_1_l2896_289631


namespace NUMINAMATH_CALUDE_paper_tearing_impossibility_l2896_289664

theorem paper_tearing_impossibility : ∀ n : ℕ, 
  n % 3 = 2 → 
  ¬ (∃ (sequence : ℕ → ℕ), 
    sequence 0 = 1 ∧ 
    (∀ i : ℕ, sequence (i + 1) = sequence i + 3 ∨ sequence (i + 1) = sequence i + 9) ∧
    (∃ k : ℕ, sequence k = n)) :=
by sorry

end NUMINAMATH_CALUDE_paper_tearing_impossibility_l2896_289664


namespace NUMINAMATH_CALUDE_square_difference_sum_l2896_289684

theorem square_difference_sum : 
  20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l2896_289684


namespace NUMINAMATH_CALUDE_factorial_solutions_l2896_289637

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_solutions :
  ∀ x y : ℕ, factorial x + 2^y = factorial (x + 1) ↔ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_solutions_l2896_289637


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l2896_289625

/-- The curve parameterized by (x,y) = (3t + 6, 5t - 7) can be expressed as y = (5/3)x - 17 --/
theorem curve_to_line_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 → y = (5/3) * x - 17 := by
  sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l2896_289625


namespace NUMINAMATH_CALUDE_constant_ratio_l2896_289622

/-- Two arithmetic sequences with sums of first n terms S_n and T_n -/
def arithmetic_sequences (S T : ℕ → ℝ) : Prop :=
  ∃ (a₁ d_a b₁ d_b : ℝ),
    ∀ n : ℕ, 
      S n = n / 2 * (2 * a₁ + (n - 1) * d_a) ∧
      T n = n / 2 * (2 * b₁ + (n - 1) * d_b)

/-- The product of sums equals n^3 - n for all positive n -/
def product_condition (S T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ+, S n * T n = (n : ℝ)^3 - n

/-- The main theorem: if the conditions are satisfied, then S_n / T_n is constant -/
theorem constant_ratio 
  (S T : ℕ → ℝ) 
  (h1 : arithmetic_sequences S T) 
  (h2 : product_condition S T) : 
  ∃ c : ℝ, ∀ n : ℕ+, S n / T n = c :=
sorry

end NUMINAMATH_CALUDE_constant_ratio_l2896_289622


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l2896_289620

theorem imaginary_part_of_product (i : ℂ) : 
  i * i = -1 →
  Complex.im ((1 + 2*i) * (2 - i)) = 3 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l2896_289620


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2896_289673

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2896_289673


namespace NUMINAMATH_CALUDE_tom_apple_purchase_l2896_289690

/-- The problem of determining how many kg of apples Tom purchased -/
theorem tom_apple_purchase (apple_price mango_price : ℕ) (mango_amount total_paid : ℕ) 
  (h1 : apple_price = 70)
  (h2 : mango_price = 70)
  (h3 : mango_amount = 9)
  (h4 : total_paid = 1190) :
  ∃ (apple_amount : ℕ), apple_amount * apple_price + mango_amount * mango_price = total_paid ∧ apple_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_apple_purchase_l2896_289690


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_foci_parabola_equation_l2896_289616

-- Define the hyperbola and ellipse
def hyperbola (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 - y^2 = m}
def ellipse := {(x, y) : ℝ × ℝ | 2*x^2 + 3*y^2 = 72}

-- Define the condition of same foci
def same_foci (h : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)) : Prop := sorry

-- Define a parabola
def parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}

-- Define the condition for focus on positive x-axis
def focus_on_positive_x (p : Set (ℝ × ℝ)) : Prop := sorry

-- Define the condition for passing through a point
def passes_through (p : Set (ℝ × ℝ)) (point : ℝ × ℝ) : Prop := point ∈ p

-- Theorem 1
theorem hyperbola_ellipse_foci (m : ℝ) : 
  same_foci (hyperbola m) ellipse → m = 6 := sorry

-- Theorem 2
theorem parabola_equation : 
  ∃ p : ℝ, focus_on_positive_x (parabola p) ∧ 
  passes_through (parabola p) (2, -4) ∧ 
  p = 4 := sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_foci_parabola_equation_l2896_289616


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l2896_289645

/-- Define a Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 0 = 0 ∧ F 1 = 1 ∧ (∀ n ≥ 2, F n = 3 * F (n - 1) - F (n - 2)) ∧ F 12 % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l2896_289645


namespace NUMINAMATH_CALUDE_sum_bc_equals_nine_l2896_289672

theorem sum_bc_equals_nine 
  (h1 : a + b = 16) 
  (h2 : c + d = 3) 
  (h3 : a + d = 10) : 
  b + c = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_bc_equals_nine_l2896_289672


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2896_289667

theorem election_votes_theorem (total_votes : ℕ) : 
  (75 : ℝ) / 100 * ((100 : ℝ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2896_289667


namespace NUMINAMATH_CALUDE_proportion_equality_l2896_289619

theorem proportion_equality (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) :
  x = 17.5 * c / (4 * b) := by
sorry

end NUMINAMATH_CALUDE_proportion_equality_l2896_289619


namespace NUMINAMATH_CALUDE_anniversary_day_theorem_l2896_289675

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years in a 300-year span -/
def leapYearsIn300Years : Nat := 73

/-- Calculates the number of regular years in a 300-year span -/
def regularYearsIn300Years : Nat := 300 - leapYearsIn300Years

/-- Calculates the total days to move backward in 300 years -/
def totalDaysBackward : Nat :=
  regularYearsIn300Years + 2 * leapYearsIn300Years

/-- Theorem: If a 300th anniversary falls on a Thursday, the original date was a Tuesday -/
theorem anniversary_day_theorem (anniversaryDay : DayOfWeek) :
  anniversaryDay = DayOfWeek.Thursday →
  (totalDaysBackward % 7 : Nat) = 2 →
  ∃ (originalDay : DayOfWeek), originalDay = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_anniversary_day_theorem_l2896_289675


namespace NUMINAMATH_CALUDE_total_marbles_l2896_289661

theorem total_marbles (dohyun_pockets : Nat) (dohyun_per_pocket : Nat)
                      (joohyun_bags : Nat) (joohyun_per_bag : Nat) :
  dohyun_pockets = 7 →
  dohyun_per_pocket = 16 →
  joohyun_bags = 6 →
  joohyun_per_bag = 25 →
  dohyun_pockets * dohyun_per_pocket + joohyun_bags * joohyun_per_bag = 262 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2896_289661


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2896_289629

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = 15) : 
  x^2 + y^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2896_289629


namespace NUMINAMATH_CALUDE_count_five_ruble_coins_l2896_289643

theorem count_five_ruble_coins 
  (total_coins : ℕ) 
  (not_two_ruble : ℕ) 
  (not_ten_ruble : ℕ) 
  (not_one_ruble : ℕ) 
  (h1 : total_coins = 25)
  (h2 : not_two_ruble = 19)
  (h3 : not_ten_ruble = 20)
  (h4 : not_one_ruble = 16) :
  total_coins - ((total_coins - not_two_ruble) + (total_coins - not_ten_ruble) + (total_coins - not_one_ruble)) = 5 := by
sorry

end NUMINAMATH_CALUDE_count_five_ruble_coins_l2896_289643


namespace NUMINAMATH_CALUDE_line_properties_l2896_289683

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (a - 1) - b = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Define point M
def point_M (a b : ℝ) : Prop := l₁ a b (-3) (-1)

-- Define equal distance from origin to both lines
def equal_distance (b : ℝ) : Prop := 4 / b = b

theorem line_properties (a b : ℝ) :
  (perpendicular a b ∧ point_M a b → a = 2 ∧ b = 2) ∧
  (parallel a b ∧ equal_distance b → (a = 2 ∧ b = -2) ∨ (a = 2/3 ∧ b = 2)) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2896_289683


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l2896_289670

/-- Given a triangle ABC with n1 points on side AB (excluding A and B),
    n2 points on side BC (excluding B and C), and n3 points on side AC (excluding A and C),
    the number of distinct triangles formed by choosing one point from each side
    is equal to n1 * n2 * n3. -/
theorem distinct_triangles_count (n1 n2 n3 : ℕ) : ℕ :=
  n1 * n2 * n3

#check distinct_triangles_count

end NUMINAMATH_CALUDE_distinct_triangles_count_l2896_289670


namespace NUMINAMATH_CALUDE_sum_difference_is_60_l2896_289618

def sum_even_2_to_120 : ℕ := (Finset.range 60).sum (fun i => 2 * (i + 1))

def sum_odd_1_to_119 : ℕ := (Finset.range 60).sum (fun i => 2 * i + 1)

theorem sum_difference_is_60 : sum_even_2_to_120 - sum_odd_1_to_119 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_60_l2896_289618


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2896_289649

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2896_289649


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2896_289636

/-- The line l is defined by the equation x + y - 1 = 0 --/
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

/-- A point P lies on line l if its coordinates satisfy the line equation --/
def point_on_line_l (x y : ℝ) : Prop := line_l x y

/-- The specific condition we're examining --/
def specific_condition (x y : ℝ) : Prop := x = 2 ∧ y = -1

theorem sufficient_not_necessary :
  (∀ x y : ℝ, specific_condition x y → point_on_line_l x y) ∧
  ¬(∀ x y : ℝ, point_on_line_l x y → specific_condition x y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2896_289636


namespace NUMINAMATH_CALUDE_rectangle_division_count_l2896_289674

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a division of a large rectangle into smaller rectangles --/
structure RectangleDivision where
  large : Rectangle
  small : Rectangle
  divisions : List (List ℕ)

/-- Counts the number of ways to divide a rectangle --/
def countDivisions (r : RectangleDivision) : ℕ :=
  r.divisions.length

/-- The main rectangle --/
def mainRectangle : Rectangle :=
  { width := 24, height := 20 }

/-- The sub-rectangle --/
def subRectangle : Rectangle :=
  { width := 5, height := 4 }

/-- The division of the main rectangle into sub-rectangles --/
def rectangleDivision : RectangleDivision :=
  { large := mainRectangle
    small := subRectangle
    divisions := [[4, 4, 4, 4, 4, 4], [4, 5, 5, 5, 5], [5, 4, 5, 5, 5], [5, 5, 4, 5, 5], [5, 5, 5, 4, 5], [5, 5, 5, 5, 4]] }

/-- Theorem stating that the number of ways to divide the rectangle is 6 --/
theorem rectangle_division_count : countDivisions rectangleDivision = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_count_l2896_289674
