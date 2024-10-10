import Mathlib

namespace number_difference_l1986_198619

theorem number_difference (x y : ℕ) : 
  x + y = 34 → 
  y = 22 → 
  y - x = 10 :=
by
  sorry

end number_difference_l1986_198619


namespace milk_conversion_theorem_l1986_198698

/-- Represents the conversion between milliliters and fluid ounces -/
structure MilkConversion where
  packets : Nat
  ml_per_packet : Nat
  total_ounces : Nat

/-- Calculates the number of milliliters in one fluid ounce -/
def ml_per_ounce (conv : MilkConversion) : Rat :=
  (conv.packets * conv.ml_per_packet) / conv.total_ounces

/-- Theorem stating that under the given conditions, one fluid ounce equals 30 ml -/
theorem milk_conversion_theorem (conv : MilkConversion) 
  (h1 : conv.packets = 150)
  (h2 : conv.ml_per_packet = 250)
  (h3 : conv.total_ounces = 1250) : 
  ml_per_ounce conv = 30 := by
  sorry

end milk_conversion_theorem_l1986_198698


namespace domain_of_f_composed_with_exp2_l1986_198690

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_f_composed_with_exp2 :
  (∀ x, f x ≠ 0 → 1 < x ∧ x < 2) →
  (∀ x, f (2^x) ≠ 0 → 0 < x ∧ x < 1) :=
sorry

end domain_of_f_composed_with_exp2_l1986_198690


namespace rectangle_max_area_l1986_198648

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → (l * w ≤ 100) :=
by
  sorry

end rectangle_max_area_l1986_198648


namespace inequality_bound_l1986_198672

theorem inequality_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) + 
  Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) < 4 := by
  sorry

end inequality_bound_l1986_198672


namespace alternating_sum_squares_l1986_198614

/-- The sum of squares with alternating signs in pairs from 1 to 120 -/
def M : ℕ → ℕ
| 0 => 0
| (n + 1) => if n % 4 < 2
              then M n + (120 - n + 1)^2
              else M n - (120 - n + 1)^2

theorem alternating_sum_squares : M 120 = 14520 := by
  sorry

end alternating_sum_squares_l1986_198614


namespace tree_height_equation_l1986_198670

/-- Represents the height of a tree over time -/
def tree_height (initial_height growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_height + growth_rate * months

/-- Theorem stating the relationship between tree height and time -/
theorem tree_height_equation (h x : ℝ) :
  h = tree_height 80 2 x ↔ h = 80 + 2 * x :=
by sorry

end tree_height_equation_l1986_198670


namespace subset_implies_a_geq_3_l1986_198608

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem subset_implies_a_geq_3 (a : ℝ) (h : A ⊆ B a) : a ≥ 3 := by
  sorry

end subset_implies_a_geq_3_l1986_198608


namespace cos_2alpha_eq_neg_one_seventh_l1986_198642

theorem cos_2alpha_eq_neg_one_seventh (α : Real) 
  (h : 3 * Real.sin (α - Real.pi/6) = Real.sin (α + Real.pi/6)) : 
  Real.cos (2 * α) = -1/7 := by
  sorry

end cos_2alpha_eq_neg_one_seventh_l1986_198642


namespace total_pears_picked_l1986_198669

theorem total_pears_picked (alyssa_pears nancy_pears : ℕ) 
  (h1 : alyssa_pears = 42) 
  (h2 : nancy_pears = 17) : 
  alyssa_pears + nancy_pears = 59 := by
  sorry

end total_pears_picked_l1986_198669


namespace complex_div_i_coords_l1986_198605

/-- The complex number (3+4i)/i corresponds to the point (4, -3) in the complex plane -/
theorem complex_div_i_coords : 
  let z : ℂ := (3 + 4*I) / I
  (z.re = 4 ∧ z.im = -3) :=
by sorry

end complex_div_i_coords_l1986_198605


namespace original_price_proof_l1986_198652

/-- Given an item sold at a 20% loss with a selling price of 1040, 
    prove that the original price of the item was 1300. -/
theorem original_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1040)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 - loss_percentage / 100) ∧ 
    original_price = 1300 :=
by
  sorry

end original_price_proof_l1986_198652


namespace probability_shortest_diagonal_decagon_l1986_198607

/-- The number of sides in a regular decagon -/
def n : ℕ := 10

/-- The total number of diagonals in a regular decagon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular decagon -/
def shortest_diagonals : ℕ := n

/-- The probability of selecting one of the shortest diagonals -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem probability_shortest_diagonal_decagon :
  probability = 2 / 7 := by sorry

end probability_shortest_diagonal_decagon_l1986_198607


namespace lars_daily_bread_production_l1986_198673

-- Define the baking rates and working hours
def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

-- Define the function to calculate total breads
def total_breads : ℕ :=
  (loaves_per_hour * hours_per_day) + 
  (baguettes_per_two_hours * (hours_per_day / 2))

-- Theorem statement
theorem lars_daily_bread_production :
  total_breads = 150 := by
  sorry

end lars_daily_bread_production_l1986_198673


namespace vlecks_for_45_degrees_l1986_198684

/-- The number of vlecks in a full circle on Venus. -/
def full_circle_vlecks : ℕ := 600

/-- The number of degrees in a full circle on Earth. -/
def full_circle_degrees : ℕ := 360

/-- Converts an angle in degrees to vlecks. -/
def degrees_to_vlecks (degrees : ℚ) : ℚ :=
  (degrees / full_circle_degrees) * full_circle_vlecks

/-- Theorem: 45 degrees corresponds to 75 vlecks on Venus. -/
theorem vlecks_for_45_degrees : degrees_to_vlecks 45 = 75 := by
  sorry

end vlecks_for_45_degrees_l1986_198684


namespace sum_of_exponents_is_eight_l1986_198661

-- Define the expression under the cube root
def radicand (x y z : ℝ) : ℝ := 40 * x^5 * y^9 * z^14

-- Define the function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (x y z : ℝ) : ℕ :=
  let simplified := (radicand x y z)^(1/3)
  -- The actual calculation of exponents would be implemented here
  -- For now, we'll use a placeholder
  8

-- Theorem statement
theorem sum_of_exponents_is_eight (x y z : ℝ) :
  sum_of_exponents_outside_radical x y z = 8 := by
  sorry

end sum_of_exponents_is_eight_l1986_198661


namespace no_valid_tiling_l1986_198699

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  rows : ℕ
  cols : ℕ

/-- Represents a domino with given dimensions -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Represents a tiling configuration -/
structure Tiling where
  rect : Rectangle
  dominos : List Domino
  count : ℕ

def is_valid_tiling (t : Tiling) : Prop :=
  t.rect.rows = 11 ∧
  t.rect.cols = 12 ∧
  t.count = 19 ∧
  ∀ d ∈ t.dominos, (d.length = 6 ∧ d.width = 1) ∨ (d.length = 7 ∧ d.width = 1) ∨
                   (d.length = 1 ∧ d.width = 6) ∨ (d.length = 1 ∧ d.width = 7)

theorem no_valid_tiling :
  ¬ ∃ t : Tiling, is_valid_tiling t := by
  sorry

end no_valid_tiling_l1986_198699


namespace gunny_bag_capacity_l1986_198663

/-- The capacity of a gunny bag filled with wheat packets -/
theorem gunny_bag_capacity
  (pounds_per_ton : ℕ)
  (ounces_per_pound : ℕ)
  (num_packets : ℕ)
  (packet_weight_pounds : ℕ)
  (packet_weight_ounces : ℕ)
  (h1 : pounds_per_ton = 2200)
  (h2 : ounces_per_pound = 16)
  (h3 : num_packets = 1760)
  (h4 : packet_weight_pounds = 16)
  (h5 : packet_weight_ounces = 4) :
  (num_packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound : ℚ)) / pounds_per_ton = 13 := by
  sorry


end gunny_bag_capacity_l1986_198663


namespace remaining_distance_l1986_198625

theorem remaining_distance (total_distance : ℝ) (father_fraction : ℝ) (mother_fraction : ℝ) :
  total_distance = 240 →
  father_fraction = 1/2 →
  mother_fraction = 3/8 →
  total_distance * (1 - father_fraction - mother_fraction) = 30 := by
sorry

end remaining_distance_l1986_198625


namespace pool_capacity_percentage_l1986_198662

/-- Calculates the current capacity percentage of a pool given its dimensions and draining parameters -/
theorem pool_capacity_percentage
  (width : ℝ) (length : ℝ) (depth : ℝ)
  (drain_rate : ℝ) (drain_time : ℝ)
  (h_width : width = 60)
  (h_length : length = 100)
  (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60)
  (h_drain_time : drain_time = 800) :
  (drain_rate * drain_time) / (width * length * depth) * 100 = 8 := by
sorry

end pool_capacity_percentage_l1986_198662


namespace gcd_lcm_product_24_36_l1986_198628

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end gcd_lcm_product_24_36_l1986_198628


namespace arithmetic_equality_l1986_198665

theorem arithmetic_equality : 142 + 29 - 32 + 25 = 164 := by sorry

end arithmetic_equality_l1986_198665


namespace fraction_difference_equals_one_minus_two_over_x_l1986_198629

theorem fraction_difference_equals_one_minus_two_over_x 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = 1 - 2 / x :=
by sorry

end fraction_difference_equals_one_minus_two_over_x_l1986_198629


namespace square_difference_sqrt5_sqrt2_l1986_198639

theorem square_difference_sqrt5_sqrt2 :
  let x : ℝ := Real.sqrt 5
  let y : ℝ := Real.sqrt 2
  (x - y)^2 = 7 - 2 * Real.sqrt 10 := by
sorry

end square_difference_sqrt5_sqrt2_l1986_198639


namespace regular_pencil_price_correct_l1986_198656

/-- The price of a regular pencil in a stationery store --/
def regular_pencil_price : ℝ :=
  let pencil_with_eraser_price : ℝ := 0.8
  let short_pencil_price : ℝ := 0.4
  let pencils_with_eraser_sold : ℕ := 200
  let regular_pencils_sold : ℕ := 40
  let short_pencils_sold : ℕ := 35
  let total_sales : ℝ := 194
  0.5

/-- Theorem stating that the regular pencil price is correct --/
theorem regular_pencil_price_correct :
  let pencil_with_eraser_price : ℝ := 0.8
  let short_pencil_price : ℝ := 0.4
  let pencils_with_eraser_sold : ℕ := 200
  let regular_pencils_sold : ℕ := 40
  let short_pencils_sold : ℕ := 35
  let total_sales : ℝ := 194
  pencil_with_eraser_price * pencils_with_eraser_sold +
  regular_pencil_price * regular_pencils_sold +
  short_pencil_price * short_pencils_sold = total_sales :=
by
  sorry

end regular_pencil_price_correct_l1986_198656


namespace function_inequality_l1986_198623

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic_2 f)
  (h_monotone : monotone_increasing_on f 0 1) :
  f (-3/2) < f (4/3) ∧ f (4/3) < f 1 :=
sorry

end function_inequality_l1986_198623


namespace honeycomb_briquettes_delivery_l1986_198602

theorem honeycomb_briquettes_delivery (total : ℕ) : 
  (3 * total) / 8 + 50 = (5 * ((total - ((3 * total) / 8 + 50)))) / 7 →
  total - ((3 * total) / 8 + 50) = 700 := by
  sorry

end honeycomb_briquettes_delivery_l1986_198602


namespace arithmetic_geometric_mean_ratio_real_l1986_198680

theorem arithmetic_geometric_mean_ratio_real (A B : ℂ) :
  (∃ r : ℝ, (A + B) / 2 = r * (A * B)^(1/2 : ℂ)) →
  (∃ r : ℝ, A = r * B) ∨ Complex.abs A = Complex.abs B :=
sorry

end arithmetic_geometric_mean_ratio_real_l1986_198680


namespace legs_on_ground_l1986_198677

theorem legs_on_ground (num_horses : ℕ) (num_men : ℕ) (num_riding : ℕ) : 
  num_horses = 8 →
  num_men = num_horses →
  num_riding = num_men / 2 →
  (4 * num_horses + 2 * (num_men - num_riding)) = 40 :=
by sorry

end legs_on_ground_l1986_198677


namespace count_perfect_square_factors_l1986_198622

/-- The number of factors of 1200 that are perfect squares -/
def perfect_square_factors : ℕ :=
  let n := 1200
  let prime_factorization := (2, 4) :: (3, 1) :: (5, 2) :: []
  sorry

/-- Theorem stating that the number of factors of 1200 that are perfect squares is 6 -/
theorem count_perfect_square_factors :
  perfect_square_factors = 6 := by sorry

end count_perfect_square_factors_l1986_198622


namespace imaginary_part_of_complex_fraction_l1986_198678

/-- The imaginary part of (1-i)^2 / (1+i) is -1 -/
theorem imaginary_part_of_complex_fraction : Complex.im ((1 - Complex.I)^2 / (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l1986_198678


namespace M_properties_M_remainder_l1986_198612

/-- The greatest integer multiple of 16 with no repeated digits -/
def M : ℕ :=
  sorry

/-- Predicate to check if a natural number has no repeated digits -/
def has_no_repeated_digits (n : ℕ) : Prop :=
  sorry

theorem M_properties :
  M % 16 = 0 ∧
  has_no_repeated_digits M ∧
  ∀ n : ℕ, n % 16 = 0 → has_no_repeated_digits n → n ≤ M :=
sorry

theorem M_remainder :
  M % 1000 = 864 :=
sorry

end M_properties_M_remainder_l1986_198612


namespace a_equals_one_sufficient_not_necessary_l1986_198674

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x y, l ≤ x → x < y → y ≤ r → f x < f y

-- State the theorem
theorem a_equals_one_sufficient_not_necessary :
  (∀ x y, 2 ≤ x → x < y → is_increasing_on (f 1) 2 y) ∧
  ¬(∀ a : ℝ, (∀ x y, 2 ≤ x → x < y → is_increasing_on (f a) 2 y) → a = 1) :=
sorry

end a_equals_one_sufficient_not_necessary_l1986_198674


namespace min_value_quadratic_l1986_198630

theorem min_value_quadratic (x : ℝ) :
  x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) →
  (x^2 + 2*x + 1) ≥ 0 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), y^2 + 2*y + 1 = 0 :=
by sorry

end min_value_quadratic_l1986_198630


namespace line_up_five_people_youngest_not_ends_l1986_198621

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_youngest_at_ends (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem line_up_five_people_youngest_not_ends : 
  number_of_arrangements 5 - arrangements_with_youngest_at_ends 5 = 72 := by
  sorry

end line_up_five_people_youngest_not_ends_l1986_198621


namespace age_difference_proof_l1986_198664

/-- The age difference between Mandy and Sarah --/
def age_difference : ℕ := by sorry

theorem age_difference_proof (mandy_age tom_age julia_age max_age sarah_age : ℕ) 
  (h1 : mandy_age = 3)
  (h2 : tom_age = 4 * mandy_age)
  (h3 : julia_age = tom_age - 5)
  (h4 : max_age = 2 * julia_age)
  (h5 : sarah_age = 3 * max_age - 1) :
  sarah_age - mandy_age = age_difference := by sorry

end age_difference_proof_l1986_198664


namespace first_nonzero_digit_after_decimal_1_197_l1986_198681

theorem first_nonzero_digit_after_decimal_1_197 : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (1000 : ℚ) / 197 = (5 : ℚ) + (d : ℚ) / (10 : ℚ) ^ (n + 1) + (1 : ℚ) / (10 : ℚ) ^ (n + 2) := by
  sorry

end first_nonzero_digit_after_decimal_1_197_l1986_198681


namespace smallest_k_for_sum_squares_multiple_of_360_l1986_198679

theorem smallest_k_for_sum_squares_multiple_of_360 :
  ∃ k : ℕ+, (k.val * (k.val + 1) * (2 * k.val + 1)) % 2160 = 0 ∧
  ∀ m : ℕ+, m < k → (m.val * (m.val + 1) * (2 * m.val + 1)) % 2160 ≠ 0 ∧
  k = 175 := by
  sorry

end smallest_k_for_sum_squares_multiple_of_360_l1986_198679


namespace decagon_diagonals_l1986_198676

/-- A convex decagon is a polygon with 10 sides -/
def ConvexDecagon : Type := Unit

/-- Number of sides in a convex decagon -/
def numSides : ℕ := 10

/-- Number of right angles in the given decagon -/
def numRightAngles : ℕ := 3

/-- The number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem decagon_diagonals (d : ConvexDecagon) : 
  numDiagonals numSides = 35 := by sorry

end decagon_diagonals_l1986_198676


namespace interval_constraint_l1986_198653

theorem interval_constraint (x : ℝ) : (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) := by
  sorry

end interval_constraint_l1986_198653


namespace tesseract_hypervolume_l1986_198686

/-- Given a tesseract with face volumes 72, 75, 48, and 50 cubic units,
    its hyper-volume is 3600 hyper-cubic units. -/
theorem tesseract_hypervolume (a b c d : ℝ) 
    (h1 : a * b * c = 72)
    (h2 : b * c * d = 75)
    (h3 : c * d * a = 48)
    (h4 : d * a * b = 50) : 
    a * b * c * d = 3600 := by
  sorry

#check tesseract_hypervolume

end tesseract_hypervolume_l1986_198686


namespace farmland_equations_correct_l1986_198645

/-- Represents the farmland purchase problem -/
structure FarmlandProblem where
  total_acres : ℕ
  good_cost_per_acre : ℚ
  bad_cost_per_seven_acres : ℚ
  total_spent : ℚ

/-- Represents the system of equations for the farmland problem -/
def farmland_equations (p : FarmlandProblem) (x y : ℚ) : Prop :=
  x + y = p.total_acres ∧
  p.good_cost_per_acre * x + (p.bad_cost_per_seven_acres / 7) * y = p.total_spent

/-- Theorem stating that the system of equations correctly represents the farmland problem -/
theorem farmland_equations_correct (p : FarmlandProblem) (x y : ℚ) :
  p.total_acres = 100 →
  p.good_cost_per_acre = 300 →
  p.bad_cost_per_seven_acres = 500 →
  p.total_spent = 10000 →
  farmland_equations p x y ↔
    (x + y = 100 ∧ 300 * x + (500 / 7) * y = 10000) :=
by sorry

end farmland_equations_correct_l1986_198645


namespace bucket_problem_l1986_198638

/-- Represents the capacity of a bucket --/
structure Bucket where
  capacity : ℝ
  sand : ℝ

/-- Proves that given the conditions of the bucket problem, 
    the initial fraction of Bucket B's capacity filled with sand is 3/8 --/
theorem bucket_problem (bucketA bucketB : Bucket) : 
  bucketA.sand = (1/4) * bucketA.capacity →
  bucketB.capacity = (1/2) * bucketA.capacity →
  bucketA.sand + bucketB.sand = (0.4375) * bucketA.capacity →
  bucketB.sand / bucketB.capacity = 3/8 := by
  sorry

end bucket_problem_l1986_198638


namespace pet_store_cats_sold_l1986_198615

theorem pet_store_cats_sold (dogs : ℕ) (cats : ℕ) : 
  cats = 3 * dogs →
  cats = 2 * (dogs + 8) →
  cats = 48 := by
sorry

end pet_store_cats_sold_l1986_198615


namespace sum_of_squares_16_to_30_l1986_198635

theorem sum_of_squares_16_to_30 :
  let sum_squares : (n : ℕ) → ℕ := λ n => n * (n + 1) * (2 * n + 1) / 6
  let sum_1_to_15 := 1280
  let sum_1_to_30 := sum_squares 30
  sum_1_to_30 - sum_1_to_15 = 8215 :=
by sorry

end sum_of_squares_16_to_30_l1986_198635


namespace interest_rate_is_nine_percent_l1986_198604

/-- Calculates the simple interest rate given two loans and the total interest received. -/
def calculate_interest_rate (principal1 : ℚ) (time1 : ℚ) (principal2 : ℚ) (time2 : ℚ) (total_interest : ℚ) : ℚ :=
  (100 * total_interest) / (principal1 * time1 + principal2 * time2)

/-- Theorem stating that the interest rate is 9% for the given loan conditions. -/
theorem interest_rate_is_nine_percent :
  let principal1 : ℚ := 5000
  let time1 : ℚ := 2
  let principal2 : ℚ := 3000
  let time2 : ℚ := 4
  let total_interest : ℚ := 1980
  calculate_interest_rate principal1 time1 principal2 time2 total_interest = 9 := by
  sorry

#eval calculate_interest_rate 5000 2 3000 4 1980

end interest_rate_is_nine_percent_l1986_198604


namespace two_digit_three_digit_sum_l1986_198631

theorem two_digit_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ 
  100000 * x + y = 7 * x * y ∧ 
  x + y = 18 := by
sorry

end two_digit_three_digit_sum_l1986_198631


namespace square_digit_sequence_l1986_198616

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def form_number (x y : ℕ) (n : ℕ) : ℕ :=
  x * (10^(2*n+1) - 10^(n+1)) / 9 + 6 * 10^n + y * (10^n - 1) / 9

theorem square_digit_sequence (x y : ℕ) : x ≠ 0 →
  (∀ n : ℕ, n ≥ 1 → is_perfect_square (form_number x y n)) →
  ((x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0)) :=
sorry

end square_digit_sequence_l1986_198616


namespace leila_payment_l1986_198644

/-- The total cost of Leila's cake order --/
def total_cost (chocolate_cakes strawberry_cakes : ℕ) 
               (chocolate_price strawberry_price : ℚ) : ℚ :=
  chocolate_cakes * chocolate_price + strawberry_cakes * strawberry_price

/-- Theorem stating that Leila should pay $168 for her cake order --/
theorem leila_payment : 
  total_cost 3 6 12 22 = 168 := by sorry

end leila_payment_l1986_198644


namespace sufficient_but_not_necessary_l1986_198668

/-- Determines if the equation x²/(k-4) - y²/(k+4) = 1 represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop := (k - 4) * (k + 4) > 0

theorem sufficient_but_not_necessary :
  (∀ k : ℝ, k ≤ -5 → is_hyperbola k) ∧
  (∃ k : ℝ, k > -5 ∧ is_hyperbola k) :=
sorry

end sufficient_but_not_necessary_l1986_198668


namespace paco_initial_sweet_cookies_l1986_198685

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := sorry

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The number of sweet cookies Paco had left -/
def remaining_sweet_cookies : ℕ := 7

/-- Theorem: Paco had 22 sweet cookies initially -/
theorem paco_initial_sweet_cookies :
  initial_sweet_cookies = eaten_sweet_cookies + remaining_sweet_cookies ∧
  initial_sweet_cookies = 22 :=
by sorry

end paco_initial_sweet_cookies_l1986_198685


namespace inscribed_circle_square_area_l1986_198634

theorem inscribed_circle_square_area (s : ℝ) (r : ℝ) : 
  r > 0 → s = 2 * r → r^2 * Real.pi = 9 * Real.pi → s^2 = 36 := by
  sorry

end inscribed_circle_square_area_l1986_198634


namespace kangaroo_equality_days_l1986_198689

/-- The number of days it takes for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos bert_kangaroos bert_daily_rate : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_daily_rate

/-- Theorem stating that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem kangaroo_equality_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

#eval days_to_equal_kangaroos 100 20 2

end kangaroo_equality_days_l1986_198689


namespace polynomial_sqrt_value_l1986_198641

theorem polynomial_sqrt_value (a₃ a₂ a₁ a₀ : ℝ) :
  let P : ℝ → ℝ := fun x ↦ x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀
  (P 1 = 1) → (P 2 = 2) → (P 3 = 3) → (P 4 = 4) →
  Real.sqrt (P 13 - 12) = 109 := by
sorry

end polynomial_sqrt_value_l1986_198641


namespace calvins_weight_loss_l1986_198637

/-- Calvin's weight loss problem -/
theorem calvins_weight_loss 
  (initial_weight : ℕ) 
  (weight_loss_per_month : ℕ) 
  (months : ℕ) 
  (h1 : initial_weight = 250)
  (h2 : weight_loss_per_month = 8)
  (h3 : months = 12) :
  initial_weight - (weight_loss_per_month * months) = 154 :=
by sorry

end calvins_weight_loss_l1986_198637


namespace cookie_problem_l1986_198683

theorem cookie_problem (tom mike millie lucy frank : ℕ) : 
  tom = 16 →
  lucy * lucy = tom →
  millie = 2 * lucy →
  mike = 3 * millie →
  frank = mike / 2 - 3 →
  frank = 9 :=
by sorry

end cookie_problem_l1986_198683


namespace atomic_weight_Br_l1986_198682

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := 137.33

/-- The molecular weight of the compound -/
def molecular_weight : ℝ := 297

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def num_Br : ℕ := 2

/-- Theorem: The atomic weight of Bromine (Br) is 79.835 -/
theorem atomic_weight_Br :
  let x := (molecular_weight - num_Ba * atomic_weight_Ba) / num_Br
  x = 79.835 := by sorry

end atomic_weight_Br_l1986_198682


namespace sequence_prime_properties_l1986_198658

/-- The sequence a(n) = 3^(2^n) + 1 for n ≥ 1 -/
def a (n : ℕ) : ℕ := 3^(2^n) + 1

/-- The set of primes that do not divide any term of the sequence -/
def nondividing_primes : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ n, n ≥ 1 → ¬(p ∣ a n)}

/-- The set of primes that divide at least one term of the sequence -/
def dividing_primes : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ n, n ≥ 1 ∧ p ∣ a n}

theorem sequence_prime_properties :
  (Set.Infinite nondividing_primes) ∧ (Set.Infinite dividing_primes) := by
  sorry

end sequence_prime_properties_l1986_198658


namespace square_51_and_39_l1986_198647

theorem square_51_and_39 : 51^2 = 2601 ∧ 39^2 = 1521 := by
  -- Given: (a ± b)² = a² ± 2ab + b²
  sorry


end square_51_and_39_l1986_198647


namespace system_solution_sum_of_squares_l1986_198617

theorem system_solution_sum_of_squares (x y : ℝ) : 
  x * y = 6 → x^2 * y + x * y^2 + x + y = 63 → x^2 + y^2 = 69 := by
  sorry

end system_solution_sum_of_squares_l1986_198617


namespace third_studio_students_l1986_198613

theorem third_studio_students (total : ℕ) (first : ℕ) (second : ℕ) 
  (h_total : total = 376)
  (h_first : first = 110)
  (h_second : second = 135) :
  total - first - second = 131 := by
  sorry

end third_studio_students_l1986_198613


namespace hexagon_square_side_ratio_l1986_198606

/-- Given a regular hexagon and a square with the same perimeter,
    this theorem proves that the ratio of the side length of the hexagon
    to the side length of the square is 2/3. -/
theorem hexagon_square_side_ratio (perimeter : ℝ) (h s : ℝ)
  (hexagon_perimeter : 6 * h = perimeter)
  (square_perimeter : 4 * s = perimeter)
  (positive_perimeter : perimeter > 0) :
  h / s = 2 / 3 :=
by sorry

end hexagon_square_side_ratio_l1986_198606


namespace pawsitive_training_center_dogs_l1986_198671

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  fetch : ℕ
  roll_over : ℕ
  sit_stay : ℕ
  sit_fetch : ℕ
  sit_roll : ℕ
  stay_fetch : ℕ
  stay_roll : ℕ
  fetch_roll : ℕ
  sit_stay_fetch : ℕ
  sit_stay_roll : ℕ
  sit_fetch_roll : ℕ
  stay_fetch_roll : ℕ
  all_four : ℕ
  none : ℕ

/-- Calculates the total number of dogs at the Pawsitive Training Center -/
def total_dogs (d : DogTricks) : ℕ := sorry

/-- Theorem stating that given the conditions, the total number of dogs is 135 -/
theorem pawsitive_training_center_dogs :
  let d : DogTricks := {
    sit := 60, stay := 35, fetch := 45, roll_over := 40,
    sit_stay := 20, sit_fetch := 15, sit_roll := 10,
    stay_fetch := 5, stay_roll := 8, fetch_roll := 6,
    sit_stay_fetch := 4, sit_stay_roll := 3,
    sit_fetch_roll := 2, stay_fetch_roll := 1,
    all_four := 2, none := 12
  }
  total_dogs d = 135 := by sorry

end pawsitive_training_center_dogs_l1986_198671


namespace smallest_n_for_g_with_large_digit_l1986_198633

/-- Sum of digits in base b representation of n -/
def digitSum (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-5 representation of n -/
def f (n : ℕ) : ℕ := digitSum n 5

/-- g(n) is the sum of digits in base-9 representation of f(n) -/
def g (n : ℕ) : ℕ := digitSum (f n) 9

/-- Checks if a number can be represented in base-17 using only digits 0-9 -/
def hasOnlyDigits0To9InBase17 (n : ℕ) : Prop := sorry

theorem smallest_n_for_g_with_large_digit : 
  (∀ m < 791, hasOnlyDigits0To9InBase17 (g m)) ∧ 
  ¬hasOnlyDigits0To9InBase17 (g 791) := by sorry

end smallest_n_for_g_with_large_digit_l1986_198633


namespace unique_base_twelve_l1986_198655

/-- Convert a base-b number to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + b * acc) 0

/-- Check if all digits in a list are less than a given base -/
def valid_digits (digits : List Nat) (b : Nat) : Prop :=
  digits.all (· < b)

/-- The main theorem statement -/
theorem unique_base_twelve :
  ∃! b : Nat, 
    b > 1 ∧
    valid_digits [3, 0, 6] b ∧
    valid_digits [4, 2, 9] b ∧
    valid_digits [7, 4, 3] b ∧
    to_decimal [3, 0, 6] b + to_decimal [4, 2, 9] b = to_decimal [7, 4, 3] b :=
by
  sorry

end unique_base_twelve_l1986_198655


namespace largest_number_in_set_l1986_198660

theorem largest_number_in_set : 
  let S : Set ℝ := {0.01, 0.2, 0.03, 0.02, 0.1}
  ∀ x ∈ S, x ≤ 0.2 ∧ 0.2 ∈ S := by
  sorry

end largest_number_in_set_l1986_198660


namespace right_triangle_area_l1986_198610

/-- Given a right triangle with circumscribed circle radius R and inscribed circle radius r,
    prove that its area is r(2R + r). -/
theorem right_triangle_area (R r : ℝ) (h_positive_R : R > 0) (h_positive_r : r > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c = 2 * R ∧
    r = (a + b - c) / 2 ∧
    a^2 + b^2 = c^2 ∧
    (1/2) * a * b = r * (2 * R + r) := by
  sorry

end right_triangle_area_l1986_198610


namespace unknown_number_multiplication_l1986_198657

theorem unknown_number_multiplication (x : ℤ) : 
  55 = x + 45 - 62 → 7 * x = 504 := by
sorry

end unknown_number_multiplication_l1986_198657


namespace complex_product_in_first_quadrant_l1986_198646

/-- The point corresponding to (1+3i)(3-i) is located in the first quadrant. -/
theorem complex_product_in_first_quadrant :
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_product_in_first_quadrant_l1986_198646


namespace sqrt_x_plus_2_meaningful_l1986_198636

theorem sqrt_x_plus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by
  sorry

end sqrt_x_plus_2_meaningful_l1986_198636


namespace john_total_cost_l1986_198687

def base_cost : ℝ := 25
def text_cost_per_message : ℝ := 0.1
def extra_minute_cost : ℝ := 0.15
def included_hours : ℝ := 20
def john_messages : ℕ := 150
def john_hours : ℝ := 22

def calculate_total_cost : ℝ :=
  base_cost +
  (↑john_messages * text_cost_per_message) +
  ((john_hours - included_hours) * 60 * extra_minute_cost)

theorem john_total_cost :
  calculate_total_cost = 58 :=
sorry

end john_total_cost_l1986_198687


namespace smallest_number_of_blocks_l1986_198627

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : Nat
  height : Nat

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : Nat
  possibleLengths : List Nat

/-- Represents the constraints for building the wall --/
structure WallConstraints where
  noCutting : Bool
  staggeredJoints : Bool
  evenEnds : Bool

/-- Calculates the smallest number of blocks needed to build the wall --/
def minBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) (constraints : WallConstraints) : Nat :=
  sorry

/-- Theorem stating the smallest number of blocks needed for the given wall --/
theorem smallest_number_of_blocks
  (wall : WallDimensions)
  (block : BlockDimensions)
  (constraints : WallConstraints)
  (h_wall_length : wall.length = 120)
  (h_wall_height : wall.height = 7)
  (h_block_height : block.height = 1)
  (h_block_lengths : block.possibleLengths = [2, 3])
  (h_no_cutting : constraints.noCutting = true)
  (h_staggered : constraints.staggeredJoints = true)
  (h_even_ends : constraints.evenEnds = true) :
  minBlocksNeeded wall block constraints = 357 :=
sorry

end smallest_number_of_blocks_l1986_198627


namespace fraction_inequality_l1986_198688

theorem fraction_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : m > 0) :
  (a + m) / (b + m) > a / b := by
  sorry

end fraction_inequality_l1986_198688


namespace investment_percentage_l1986_198603

/-- Proves that given a sum of 4000 Rs invested at 18% p.a. for two years yields 480 Rs more in interest
    than if it were invested at x% p.a. for the same period, x must equal 12%. -/
theorem investment_percentage (x : ℝ) : 
  (4000 * 18 * 2 / 100 - 4000 * x * 2 / 100 = 480) → x = 12 := by
  sorry

end investment_percentage_l1986_198603


namespace star_difference_sum_l1986_198600

/-- The ⋆ operation for real numbers -/
def star (a b : ℝ) : ℝ := a^2 - b

/-- Theorem stating the result of (x - y) ⋆ (x + y) -/
theorem star_difference_sum (x y : ℝ) : 
  star (x - y) (x + y) = x^2 - x - 2*x*y + y^2 - y := by
  sorry

end star_difference_sum_l1986_198600


namespace shells_in_morning_l1986_198659

theorem shells_in_morning (afternoon_shells : ℕ) (total_shells : ℕ) 
  (h1 : afternoon_shells = 324)
  (h2 : total_shells = 616) :
  total_shells - afternoon_shells = 292 := by
  sorry

end shells_in_morning_l1986_198659


namespace a_fourth_plus_reciprocal_l1986_198675

theorem a_fourth_plus_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/a^4 = 7 := by
  sorry

end a_fourth_plus_reciprocal_l1986_198675


namespace simplify_absolute_expression_l1986_198649

theorem simplify_absolute_expression : |(-4^2 + 6^2 - 2)| = 18 := by
  sorry

end simplify_absolute_expression_l1986_198649


namespace hoseok_number_problem_l1986_198697

theorem hoseok_number_problem (x : ℝ) : 15 * x = 45 → x - 1 = 2 := by
  sorry

end hoseok_number_problem_l1986_198697


namespace blue_marbles_count_l1986_198618

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 20 →
  red = 9 →
  prob_red_or_white = 7/10 →
  ∃ (blue white : ℕ),
    blue + red + white = total ∧
    (red + white : ℚ) / total = prob_red_or_white ∧
    blue = 6 :=
by
  sorry

end blue_marbles_count_l1986_198618


namespace valentines_day_theorem_l1986_198696

theorem valentines_day_theorem (x y : ℕ) : 
  x * y = x + y + 28 → x * y = 60 :=
by sorry

end valentines_day_theorem_l1986_198696


namespace cone_slant_height_l1986_198611

/-- Represents the properties of a cone --/
structure Cone where
  baseRadius : ℝ
  sectorAngle : ℝ
  slantHeight : ℝ

/-- Theorem: For a cone with base radius 6 cm and sector angle 240°, the slant height is 9 cm --/
theorem cone_slant_height (c : Cone) 
  (h1 : c.baseRadius = 6)
  (h2 : c.sectorAngle = 240) : 
  c.slantHeight = 9 := by
  sorry

#check cone_slant_height

end cone_slant_height_l1986_198611


namespace germination_probability_convergence_l1986_198667

/-- Represents the experimental data for rice seed germination --/
structure GerminationData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations
  h : m ≤ n

/-- The list of experimental data --/
def experimentalData : List GerminationData := [
  ⟨50, 47, sorry⟩,
  ⟨100, 89, sorry⟩,
  ⟨200, 188, sorry⟩,
  ⟨500, 461, sorry⟩,
  ⟨1000, 892, sorry⟩,
  ⟨2000, 1826, sorry⟩,
  ⟨3000, 2733, sorry⟩
]

/-- The germination frequency for a given experiment --/
def germinationFrequency (data : GerminationData) : ℚ :=
  data.m / data.n

/-- The estimated probability of germination --/
def estimatedProbability : ℚ := 91 / 100

/-- Theorem stating that the germination frequency approaches the estimated probability as sample size increases --/
theorem germination_probability_convergence :
  ∀ ε > 0, ∃ N, ∀ data ∈ experimentalData, data.n ≥ N →
    |germinationFrequency data - estimatedProbability| < ε :=
sorry

end germination_probability_convergence_l1986_198667


namespace trajectory_max_value_l1986_198694

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x + 1)^2 + (4/3) * y^2 = 4

/-- The distance ratio condition -/
def distance_ratio (x y : ℝ) : Prop :=
  (x^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

theorem trajectory_max_value :
  ∀ x y : ℝ, 
    distance_ratio x y → 
    trajectory x y → 
    2 * x^2 + y^2 ≤ 18 :=
sorry

end trajectory_max_value_l1986_198694


namespace function_properties_l1986_198651

/-- Given a function f(x) = x + m/x where f(1) = 5, this theorem proves:
    1. The value of m
    2. The parity of f
    3. The monotonicity of f on (2, +∞) -/
theorem function_properties (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x ≠ 0, f x = x + m / x)
    (h2 : f 1 = 5) :
    (m = 4) ∧ 
    (∀ x ≠ 0, f (-x) = -f x) ∧
    (∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end function_properties_l1986_198651


namespace sum_of_x_values_l1986_198620

theorem sum_of_x_values (N : ℝ) (h : N ≥ 0) : 
  ∃ x₁ x₂ : ℝ, |x₁ - 25| = N ∧ |x₂ - 25| = N ∧ x₁ + x₂ = 50 :=
by sorry

end sum_of_x_values_l1986_198620


namespace max_abs_z_on_circle_l1986_198650

open Complex

theorem max_abs_z_on_circle (z : ℂ) (h : abs (z - (3 + 4*I)) = 1) : 
  (∀ w : ℂ, abs (w - (3 + 4*I)) = 1 → abs w ≤ abs z) → abs z = 6 :=
sorry

end max_abs_z_on_circle_l1986_198650


namespace rectangular_box_volume_l1986_198643

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  let volume := x * (2 * x) * (5 * x)
  (volume = 80 ∨ volume = 250 ∨ volume = 500 ∨ volume = 1000 ∨ volume = 2000) →
  volume = 80 :=
by sorry

end rectangular_box_volume_l1986_198643


namespace function_difference_l1986_198640

theorem function_difference (k : ℝ) : 
  let f (x : ℝ) := 4 * x^2 - 3 * x + 5
  let g (x : ℝ) := 2 * x^2 - k * x + 1
  (f 10 - g 10 = 20) → k = -21.4 := by
sorry

end function_difference_l1986_198640


namespace simple_interest_rate_calculation_l1986_198626

/-- Given that a sum of money becomes 7/6 of itself in 2 years under simple interest,
    prove that the rate of interest per annum is 100/12. -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R = 100 / 12 ∧ P * (1 + R * 2 / 100) = 7 / 6 * P :=
by sorry

end simple_interest_rate_calculation_l1986_198626


namespace polynomial_divisibility_l1986_198695

theorem polynomial_divisibility (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  (x - y)^5 + (y - z)^5 + (z - x)^5 = 
  -5 * (x - y) * (y - z) * (z - x) * ((x - y)^2 + (x - y) * (y - z) + (y - z)^2) :=
by sorry

end polynomial_divisibility_l1986_198695


namespace min_value_x_plus_y_l1986_198624

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 6*y - x*y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + 6*b - a*b = 0 → x + y ≤ a + b ∧ x + y = 8 + 4*Real.sqrt 3 :=
sorry

end min_value_x_plus_y_l1986_198624


namespace triangle_angle_solution_l1986_198601

theorem triangle_angle_solution (angle1 angle2 angle3 : ℝ) (x : ℝ) : 
  angle1 = 40 ∧ 
  angle2 = 4 * x ∧ 
  angle3 = 3 * x ∧ 
  angle1 + angle2 + angle3 = 180 →
  x = 20 := by
sorry

end triangle_angle_solution_l1986_198601


namespace mitchell_unchewed_gum_l1986_198632

theorem mitchell_unchewed_gum (packets : ℕ) (pieces_per_packet : ℕ) (chewed_pieces : ℕ) 
  (h1 : packets = 8) 
  (h2 : pieces_per_packet = 7) 
  (h3 : chewed_pieces = 54) : 
  packets * pieces_per_packet - chewed_pieces = 2 := by
  sorry

end mitchell_unchewed_gum_l1986_198632


namespace earloop_probability_is_0_12_l1986_198666

/-- Represents a mask factory with two types of products -/
structure MaskFactory where
  regularProportion : ℝ
  surgicalProportion : ℝ
  regularEarloopProportion : ℝ
  surgicalEarloopProportion : ℝ

/-- The probability of selecting a mask with ear loops from the factory -/
def earloopProbability (factory : MaskFactory) : ℝ :=
  factory.regularProportion * factory.regularEarloopProportion +
  factory.surgicalProportion * factory.surgicalEarloopProportion

/-- Theorem stating the probability of selecting a mask with ear loops -/
theorem earloop_probability_is_0_12 (factory : MaskFactory)
  (h1 : factory.regularProportion = 0.8)
  (h2 : factory.surgicalProportion = 0.2)
  (h3 : factory.regularEarloopProportion = 0.1)
  (h4 : factory.surgicalEarloopProportion = 0.2) :
  earloopProbability factory = 0.12 := by
  sorry


end earloop_probability_is_0_12_l1986_198666


namespace peanuts_in_box_l1986_198693

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 4

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 8

/-- The total number of peanuts in the box after Mary adds more -/
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_in_box : total_peanuts = 12 := by
  sorry

end peanuts_in_box_l1986_198693


namespace point_three_units_away_l1986_198692

theorem point_three_units_away (A : ℝ) (h : A = 2) :
  ∀ B : ℝ, abs (B - A) = 3 → (B = -1 ∨ B = 5) :=
by sorry

end point_three_units_away_l1986_198692


namespace quadratic_minimum_value_l1986_198654

/-- The minimum value of a quadratic function -/
theorem quadratic_minimum_value 
  (p q r : ℝ) 
  (h1 : p > 0) 
  (h2 : q^2 - 4*p*r < 0) : 
  ∃ (x : ℝ), ∀ (y : ℝ), p*y^2 + q*y + r ≥ (4*p*r - q^2) / (4*p) :=
sorry

end quadratic_minimum_value_l1986_198654


namespace hyperbola_triangle_area_l1986_198691

/-- The hyperbola with equation x^2/9 - y^2/16 = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) - (p.2^2 / 16) = 1}

/-- The right focus of the hyperbola -/
def F : ℝ × ℝ := (5, 0)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- A point on the hyperbola where a line perpendicular to an asymptote intersects it -/
def P : ℝ × ℝ := sorry

/-- The area of triangle OPF -/
def area_OPF : ℝ := sorry

theorem hyperbola_triangle_area :
  area_OPF = 6 := by sorry

end hyperbola_triangle_area_l1986_198691


namespace diff_sums_1500_l1986_198609

/-- Sum of the first n odd natural numbers -/
def sumOddNaturals (n : ℕ) : ℕ := n * n

/-- Sum of the first n even natural numbers -/
def sumEvenNaturals (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even natural numbers (each reduced by 3) 
    and the sum of the first n odd natural numbers -/
def diffSums (n : ℕ) : ℤ :=
  (sumEvenNaturals n - 3 * n : ℤ) - sumOddNaturals n

theorem diff_sums_1500 : diffSums 1500 = -2250 := by
  sorry

#eval diffSums 1500

end diff_sums_1500_l1986_198609
