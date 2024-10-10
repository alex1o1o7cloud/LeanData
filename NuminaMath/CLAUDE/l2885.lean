import Mathlib

namespace indefinite_integral_proof_l2885_288561

theorem indefinite_integral_proof (x : ℝ) : 
  (deriv (fun x => -1/2 * (2 - 3*x) * Real.cos (2*x) - 3/4 * Real.sin (2*x))) x 
  = (2 - 3*x) * Real.sin (2*x) := by
sorry

end indefinite_integral_proof_l2885_288561


namespace nine_sided_polygon_diagonals_l2885_288521

/-- The number of diagonals in a regular polygon with n sides -/
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l2885_288521


namespace total_money_made_l2885_288515

/-- The total money made from selling items is the sum of the products of price and quantity for each item. -/
theorem total_money_made 
  (smoothie_price cake_price : ℚ) 
  (smoothie_quantity cake_quantity : ℕ) :
  smoothie_price * smoothie_quantity + cake_price * cake_quantity =
  (smoothie_price * smoothie_quantity + cake_price * cake_quantity : ℚ) :=
by sorry

/-- Scott's total earnings from selling smoothies and cakes -/
def scotts_earnings : ℚ :=
  let smoothie_price : ℚ := 3
  let cake_price : ℚ := 2
  let smoothie_quantity : ℕ := 40
  let cake_quantity : ℕ := 18
  smoothie_price * smoothie_quantity + cake_price * cake_quantity

#eval scotts_earnings -- Expected output: 156

end total_money_made_l2885_288515


namespace skittles_per_friend_l2885_288537

def total_skittles : ℕ := 40
def num_friends : ℕ := 5

theorem skittles_per_friend :
  total_skittles / num_friends = 8 := by sorry

end skittles_per_friend_l2885_288537


namespace subset_condition_disjoint_condition_l2885_288524

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Question 1
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Question 2
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end subset_condition_disjoint_condition_l2885_288524


namespace min_people_for_company2_cheaper_l2885_288505

/-- Company 1's pricing function -/
def company1_cost (people : ℕ) : ℕ := 150 + 18 * people

/-- Company 2's pricing function -/
def company2_cost (people : ℕ) : ℕ := 250 + 15 * people

/-- Theorem stating the minimum number of people for Company 2 to be cheaper -/
theorem min_people_for_company2_cheaper :
  (company2_cost 34 < company1_cost 34) ∧
  (company1_cost 33 ≤ company2_cost 33) := by
  sorry

end min_people_for_company2_cheaper_l2885_288505


namespace building_units_l2885_288567

theorem building_units (total : ℕ) (restaurants : ℕ) : 
  (2 * restaurants = total / 4) →
  (restaurants = 75) →
  (total = 300) := by
sorry

end building_units_l2885_288567


namespace number_equals_sixteen_l2885_288563

theorem number_equals_sixteen (x y : ℝ) (h1 : |x| = 9*x - y) (h2 : x = 2) : y = 16 := by
  sorry

end number_equals_sixteen_l2885_288563


namespace car_storm_distance_time_l2885_288562

/-- The time when a car traveling north at 3/4 mile per minute is 30 miles away from the center of a storm
    moving southeast at 3/4√2 mile per minute, given that at t=0 the storm's center is 150 miles due east of the car. -/
theorem car_storm_distance_time : ∃ t : ℝ,
  (27 / 32 : ℝ) * t^2 - (450 * Real.sqrt 2 / 2) * t + 21600 = 0 :=
by sorry

end car_storm_distance_time_l2885_288562


namespace dance_attendance_l2885_288585

theorem dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end dance_attendance_l2885_288585


namespace least_three_digit_multiple_l2885_288540

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 := by
  sorry

end least_three_digit_multiple_l2885_288540


namespace f_2013_plus_f_neg_2014_l2885_288512

open Real

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x ≥ 0, f (x + 2) = f x

def matches_exp_minus_one_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = exp x - 1

theorem f_2013_plus_f_neg_2014 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : is_periodic_2 f)
  (h_match : matches_exp_minus_one_on_interval f) :
  f 2013 + f (-2014) = exp 1 - 1 := by
  sorry

end f_2013_plus_f_neg_2014_l2885_288512


namespace product_not_equal_48_l2885_288598

theorem product_not_equal_48 : ∃! (a b : ℚ), (a, b) ∈ ({(-4, -12), (-3, -16), (1/2, -96), (1, 48), (4/3, 36)} : Set (ℚ × ℚ)) ∧ a * b ≠ 48 := by
  sorry

end product_not_equal_48_l2885_288598


namespace professors_women_tenured_or_both_l2885_288555

theorem professors_women_tenured_or_both (
  women_percentage : Real)
  (tenured_percentage : Real)
  (men_tenured_percentage : Real)
  (h1 : women_percentage = 0.69)
  (h2 : tenured_percentage = 0.70)
  (h3 : men_tenured_percentage = 0.52) :
  women_percentage + tenured_percentage - (tenured_percentage - men_tenured_percentage * (1 - women_percentage)) = 0.8512 := by
  sorry

end professors_women_tenured_or_both_l2885_288555


namespace simple_interest_investment_l2885_288522

/-- Proves that an initial investment of $1000 with 10% simple interest over 3 years results in $1300 --/
theorem simple_interest_investment (P : ℝ) : 
  (P * (1 + 0.1 * 3) = 1300) → P = 1000 := by
  sorry

end simple_interest_investment_l2885_288522


namespace intersection_trig_functions_l2885_288549

theorem intersection_trig_functions (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) 
  (h3 : 6 * Real.cos x = 9 * Real.tan x) : Real.sin x = 1 / 2 := by
  sorry

end intersection_trig_functions_l2885_288549


namespace clare_remaining_money_l2885_288542

-- Define the initial amount and item costs
def initial_amount : ℚ := 47
def bread_cost : ℚ := 2
def milk_cost : ℚ := 2
def cereal_cost : ℚ := 3
def apple_cost : ℚ := 4

-- Define the quantities of each item
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def cereal_quantity : ℕ := 3
def apple_quantity : ℕ := 1

-- Define the discount and tax rates
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining money
def calculate_remaining_money : ℚ :=
  let total_cost := bread_cost * bread_quantity + milk_cost * milk_quantity + 
                    cereal_cost * cereal_quantity + apple_cost * apple_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  let final_cost := discounted_cost + tax_amount
  initial_amount - final_cost

-- Theorem statement
theorem clare_remaining_money :
  calculate_remaining_money = 23.37 := by sorry

end clare_remaining_money_l2885_288542


namespace inequality_proof_ratio_proof_l2885_288571

-- Part I
theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by sorry

-- Part II
theorem ratio_proof (a b c x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : a^2 + b^2 + c^2 = 10) (h8 : x^2 + y^2 + z^2 = 40) (h9 : a*x + b*y + c*z = 20) :
  (a + b + c) / (x + y + z) = 1/2 := by sorry

end inequality_proof_ratio_proof_l2885_288571


namespace rational_numbers_product_sum_negative_l2885_288594

theorem rational_numbers_product_sum_negative (x y : ℚ) 
  (h_product : x * y < 0) 
  (h_sum : x + y < 0) : 
  (abs x > abs y ∧ x < 0 ∧ y > 0) ∨ (abs y > abs x ∧ y < 0 ∧ x > 0) := by
  sorry

end rational_numbers_product_sum_negative_l2885_288594


namespace cookout_2006_attendance_l2885_288501

def cookout_2004 : ℕ := 60

def cookout_2005 : ℕ := cookout_2004 / 2

def cookout_2006 : ℕ := (cookout_2005 * 2) / 3

theorem cookout_2006_attendance : cookout_2006 = 20 := by
  sorry

end cookout_2006_attendance_l2885_288501


namespace triangle_cosine_inequality_l2885_288589

theorem triangle_cosine_inequality (A B C : Real) : 
  A > 0 → B > 0 → C > 0 → A + B + C = Real.pi → 
  Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) ≤ 3 * Real.sqrt 3 / 2 ∧
  (Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) = 3 * Real.sqrt 3 / 2 ↔ 
   A = Real.pi/3 ∧ B = Real.pi/3 ∧ C = Real.pi/3) :=
by sorry

end triangle_cosine_inequality_l2885_288589


namespace natalia_crates_l2885_288535

theorem natalia_crates (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) (crate_capacity : ℕ) :
  novels = 145 →
  comics = 271 →
  documentaries = 419 →
  albums = 209 →
  crate_capacity = 9 →
  (novels + comics + documentaries + albums + crate_capacity - 1) / crate_capacity = 116 :=
by sorry

end natalia_crates_l2885_288535


namespace unique_real_roots_l2885_288509

def n : ℕ := 2016

-- Define geometric progression
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : ℕ, i < n → a (i + 1) = r * a i

-- Define arithmetic progression
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i < n → b (i + 1) = b i + d

-- Define quadratic polynomial
def P (a b : ℕ → ℝ) (i : ℕ) (x : ℝ) : ℝ :=
  x^2 + a i * x + b i

-- Define discriminant
def discriminant (a b : ℕ → ℝ) (i : ℕ) : ℝ :=
  (a i)^2 - 4 * b i

-- Theorem statement
theorem unique_real_roots
  (a : ℕ → ℝ) (b : ℕ → ℝ) (k : ℕ)
  (h_geo : is_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_unique : ∀ i : ℕ, i ≤ n → i ≠ k → discriminant a b i < 0)
  (h_real : discriminant a b k ≥ 0) :
  k = 1 ∨ k = n := by sorry

end unique_real_roots_l2885_288509


namespace trapezoid_longer_side_length_l2885_288530

/-- Given a square of side length s divided into a pentagon and three congruent trapezoids,
    if all four shapes have equal area, then the length of the longer parallel side
    of each trapezoid is s/2 -/
theorem trapezoid_longer_side_length (s : ℝ) (s_pos : s > 0) :
  let square_area := s^2
  let shape_area := square_area / 4
  let trapezoid_height := s / 2
  ∃ x : ℝ,
    x > 0 ∧
    x < s ∧
    shape_area = (x + s/2) * trapezoid_height / 2 ∧
    x = s / 2 :=
by sorry

end trapezoid_longer_side_length_l2885_288530


namespace solve_for_m_l2885_288559

/-- 
If 2x + m = 6 and x = 2, then m = 2
-/
theorem solve_for_m (x m : ℝ) (eq : 2 * x + m = 6) (sol : x = 2) : m = 2 := by
  sorry

end solve_for_m_l2885_288559


namespace line_inclination_45_degrees_l2885_288502

theorem line_inclination_45_degrees (a : ℝ) : 
  (∃ (x y : ℝ), ax + (2*a - 3)*y = 0) →   -- Line equation
  (Real.arctan (-a / (2*a - 3)) = π/4) →  -- 45° inclination
  a = 1 := by
sorry

end line_inclination_45_degrees_l2885_288502


namespace mother_double_age_in_18_years_l2885_288570

/-- Represents the number of years until Xiaoming's mother's age is twice Xiaoming's age -/
def years_until_double_age (xiaoming_age : ℕ) (mother_age : ℕ) : ℕ :=
  mother_age - 2 * xiaoming_age

theorem mother_double_age_in_18_years :
  let xiaoming_current_age : ℕ := 6
  let mother_current_age : ℕ := 30
  years_until_double_age xiaoming_current_age mother_current_age = 18 :=
by
  sorry

#check mother_double_age_in_18_years

end mother_double_age_in_18_years_l2885_288570


namespace inequality_solution_set_l2885_288539

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-1/5 : ℝ) (8/15 : ℝ)) = {x | (7/30 : ℝ) + |x - 13/60| < 11/20} :=
by sorry

end inequality_solution_set_l2885_288539


namespace distribution_count_l2885_288546

/-- Represents the number of ways to distribute items between two people. -/
def distribute (pencils notebooks pens : Nat) : Nat :=
  let pencil_distributions := 3  -- (1,3), (2,2), (3,1)
  let notebook_distributions := 1  -- (1,1)
  let pen_distributions := 2  -- (1,2), (2,1)
  pencil_distributions * notebook_distributions * pen_distributions

/-- Theorem stating that the number of ways to distribute the given items is 6. -/
theorem distribution_count :
  ∀ (erasers : Nat), erasers > 0 → distribute 4 2 3 = 6 := by
  sorry

end distribution_count_l2885_288546


namespace division_increase_by_digit_swap_l2885_288596

theorem division_increase_by_digit_swap (n : Nat) (d : Nat) :
  n = 952473 →
  d = 18 →
  (954273 / d) - (n / d) = 100 :=
by
  sorry

end division_increase_by_digit_swap_l2885_288596


namespace sum_of_ages_after_20_years_l2885_288590

/-- Given the ages of Ann and her siblings and cousin, calculate the sum of their ages after 20 years -/
theorem sum_of_ages_after_20_years 
  (ann_age : ℕ)
  (tom_age : ℕ)
  (bill_age : ℕ)
  (cathy_age : ℕ)
  (emily_age : ℕ)
  (h1 : ann_age = 6)
  (h2 : tom_age = 2 * ann_age)
  (h3 : bill_age = tom_age - 3)
  (h4 : cathy_age = 2 * tom_age)
  (h5 : emily_age = cathy_age / 2)
  : ann_age + tom_age + bill_age + cathy_age + emily_age + 20 * 5 = 163 := by
  sorry

end sum_of_ages_after_20_years_l2885_288590


namespace divisor_with_remainder_54_l2885_288503

theorem divisor_with_remainder_54 :
  ∃ (n : ℕ), n > 0 ∧ (55^55 + 55) % n = 54 ∧ n = 56 := by sorry

end divisor_with_remainder_54_l2885_288503


namespace smallest_valid_n_l2885_288583

def is_valid (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧
  (n^2 + n) % k₁ = 0 ∧ (n^2 + n) % k₂ ≠ 0

theorem smallest_valid_n :
  is_valid 4 ∧ ∀ m : ℕ, 0 < m ∧ m < 4 → ¬is_valid m :=
sorry

end smallest_valid_n_l2885_288583


namespace speed_of_current_l2885_288518

/-- Calculates the speed of the current given the rowing speed in still water,
    distance covered downstream, and time taken. -/
theorem speed_of_current
  (rowing_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : rowing_speed = 120)
  (h2 : distance = 0.5)
  (h3 : time = 9.99920006399488 / 3600) :
  rowing_speed + (distance / time - rowing_speed) = 180 :=
by sorry

#check speed_of_current

end speed_of_current_l2885_288518


namespace monster_count_monster_count_proof_l2885_288510

theorem monster_count : ℕ → Prop :=
  fun m : ℕ =>
    ∃ s : ℕ,
      s = 4 * m + 3 ∧
      s = 5 * m - 6 →
      m = 9

-- The proof is omitted
theorem monster_count_proof : monster_count 9 := by
  sorry

end monster_count_monster_count_proof_l2885_288510


namespace difference_of_squares_l2885_288525

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end difference_of_squares_l2885_288525


namespace y_is_twenty_percent_of_x_l2885_288591

/-- Given two equations involving x, y, and z, prove that y is 20% of x -/
theorem y_is_twenty_percent_of_x (x y z : ℝ) 
  (eq1 : 0.3 * (x - y) = 0.2 * (x + y))
  (eq2 : 0.4 * (x + z) = 0.1 * (y - z)) :
  y = 0.2 * x := by
  sorry

end y_is_twenty_percent_of_x_l2885_288591


namespace product_of_real_parts_l2885_288582

theorem product_of_real_parts (x : ℂ) : 
  x^2 + 4*x = -1 + Complex.I → 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ = -1 + Complex.I ∧ x₂^2 + 4*x₂ = -1 + Complex.I ∧ 
    (x₁.re * x₂.re = (1 + 3 * Real.sqrt 10) / 2)) := by
  sorry

end product_of_real_parts_l2885_288582


namespace train_distance_difference_l2885_288556

/-- Represents the distance traveled by a train given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the total distance between two points -/
def total_distance : ℝ := 900

/-- Represents the speed of the first train -/
def speed1 : ℝ := 50

/-- Represents the speed of the second train -/
def speed2 : ℝ := 40

/-- Theorem stating the difference in distance traveled by two trains -/
theorem train_distance_difference :
  ∃ (time : ℝ), 
    time > 0 ∧
    distance speed1 time + distance speed2 time = total_distance ∧
    distance speed1 time - distance speed2 time = 100 :=
sorry

end train_distance_difference_l2885_288556


namespace coconuts_yield_five_l2885_288547

/-- The number of coconuts each tree yields -/
def coconuts_per_tree (price_per_coconut : ℚ) (total_amount : ℚ) (num_trees : ℕ) : ℚ :=
  (total_amount / price_per_coconut) / num_trees

/-- Proof that each tree yields 5 coconuts given the conditions -/
theorem coconuts_yield_five :
  coconuts_per_tree 3 90 6 = 5 := by
  sorry

end coconuts_yield_five_l2885_288547


namespace no_solution_and_inequality_solution_l2885_288578

theorem no_solution_and_inequality_solution :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1) ∧
  (∀ x : ℝ, 2 * (x - 1) ≥ x + 1 ∧ x - 2 > (2 * x - 1) / 3 ↔ x > 5) :=
by sorry

end no_solution_and_inequality_solution_l2885_288578


namespace quadratic_vertex_property_l2885_288508

/-- Given a quadratic function y = -x^2 + 2x + n with vertex (m, 1), prove m - n = 1 -/
theorem quadratic_vertex_property (n m : ℝ) : 
  (∀ x, -x^2 + 2*x + n = -(x - m)^2 + 1) → m - n = 1 := by
sorry

end quadratic_vertex_property_l2885_288508


namespace fraction_simplification_l2885_288548

theorem fraction_simplification : (1922^2 - 1913^2) / (1930^2 - 1905^2) = 9 / 25 := by
  sorry

end fraction_simplification_l2885_288548


namespace geometric_sequence_third_term_l2885_288527

/-- Given a geometric sequence {a_n} where a₁ = 1 and a₅ = 81, prove that a₃ = 9 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 1) 
  (h_a5 : a 5 = 81) : 
  a 3 = 9 := by
sorry

end geometric_sequence_third_term_l2885_288527


namespace classroom_fraction_l2885_288534

theorem classroom_fraction (total : ℕ) (absent_fraction : ℚ) (canteen : ℕ) : 
  total = 40 → 
  absent_fraction = 1 / 10 → 
  canteen = 9 → 
  (total - (absent_fraction * total).num - canteen : ℚ) / (total - (absent_fraction * total).num) = 3 / 4 := by
  sorry

end classroom_fraction_l2885_288534


namespace quadratic_root_difference_l2885_288564

theorem quadratic_root_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -6 ∧ c = 8 → |r₁ - r₂| = 2 :=
by
  sorry

#check quadratic_root_difference

end quadratic_root_difference_l2885_288564


namespace silver_cube_side_length_l2885_288579

/-- Proves that a silver cube sold for $4455 at 110% of its silver value, 
    where a cubic inch of silver weighs 6 ounces and each ounce of silver 
    sells for $25, has a side length of 3 inches. -/
theorem silver_cube_side_length :
  let selling_price : ℝ := 4455
  let markup_percentage : ℝ := 1.10
  let weight_per_cubic_inch : ℝ := 6
  let price_per_ounce : ℝ := 25
  let side_length : ℝ := (selling_price / markup_percentage / price_per_ounce / weight_per_cubic_inch) ^ (1/3)
  side_length = 3 := by sorry

end silver_cube_side_length_l2885_288579


namespace hyperbola_asymptote_l2885_288599

/-- The equation of an asymptote of the hyperbola y²/8 - x²/6 = 1 -/
theorem hyperbola_asymptote :
  ∃ (x y : ℝ), (y^2 / 8 - x^2 / 6 = 1) →
  (2 * x - Real.sqrt 3 * y = 0 ∨ 2 * x + Real.sqrt 3 * y = 0) := by
  sorry

end hyperbola_asymptote_l2885_288599


namespace complex_number_quadrant_l2885_288526

/-- The complex number i(2-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Complex.I * (2 - Complex.I) = Complex.mk x y := by
  sorry

end complex_number_quadrant_l2885_288526


namespace equation_solutions_l2885_288552

theorem equation_solutions (a : ℝ) (h : a < 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧
  (∀ x ∈ s, -π < x ∧ x < π) ∧
  (∀ x ∈ s, (a - 1) * (Real.sin (2 * x) + Real.cos x) + (a + 1) * (Real.sin x - Real.cos (2 * x)) = 0) ∧
  (∀ x, -π < x ∧ x < π →
    (a - 1) * (Real.sin (2 * x) + Real.cos x) + (a + 1) * (Real.sin x - Real.cos (2 * x)) = 0 →
    x ∈ s) :=
by sorry

end equation_solutions_l2885_288552


namespace cubic_root_equation_solution_l2885_288506

theorem cubic_root_equation_solution :
  ∃ x : ℝ, x > 0 ∧ 3 * (2 + x)^(1/3) + 4 * (2 - x)^(1/3) = 6 ∧ |x - 2.096| < 0.001 := by
  sorry

end cubic_root_equation_solution_l2885_288506


namespace tyler_meal_choices_l2885_288588

-- Define the number of options for each food category
def num_meats : ℕ := 3
def num_vegetables : ℕ := 5
def num_desserts : ℕ := 4

-- Define the number of vegetables to be chosen
def vegetables_to_choose : ℕ := 2

-- Theorem statement
theorem tyler_meal_choices :
  (num_meats * (Nat.choose num_vegetables vegetables_to_choose) * num_desserts) = 120 := by
  sorry

end tyler_meal_choices_l2885_288588


namespace cube_root_simplification_l2885_288565

theorem cube_root_simplification :
  Real.rpow (20^3 + 30^3 + 40^3) (1/3) = 10 * Real.rpow 99 (1/3) :=
by sorry

end cube_root_simplification_l2885_288565


namespace extreme_value_implies_a_l2885_288532

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - (-3)| < ε → f a x ≤ f a (-3) ∨ f a x ≥ f a (-3)) →
  a = 5 :=
sorry

end extreme_value_implies_a_l2885_288532


namespace triangle_pieces_count_l2885_288584

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Number of rods in the triangle -/
def totalRods : ℕ := arithmeticSum 5 5 10

/-- Number of connectors in the triangle -/
def totalConnectors : ℕ := arithmeticSum 3 3 11

/-- Total number of pieces in the triangle -/
def totalPieces : ℕ := totalRods + totalConnectors

theorem triangle_pieces_count : totalPieces = 473 := by
  sorry

end triangle_pieces_count_l2885_288584


namespace perfect_square_condition_l2885_288517

theorem perfect_square_condition (x : ℤ) : 
  ∃ (y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
by sorry

end perfect_square_condition_l2885_288517


namespace expansion_of_binomial_l2885_288586

theorem expansion_of_binomial (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₁ + a₂ + a₃ + a₄ = -80 ∧ (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
  sorry

end expansion_of_binomial_l2885_288586


namespace distributive_law_addition_over_multiplication_not_hold_l2885_288544

-- Define the pair type
def Pair := ℝ × ℝ

-- Define addition operation
def add : Pair → Pair → Pair
  | (x₁, y₁), (x₂, y₂) => (x₁ + x₂, y₁ + y₂)

-- Define multiplication operation
def mul : Pair → Pair → Pair
  | (x₁, y₁), (x₂, y₂) => (x₁ * x₂ - y₁ * y₂, x₁ * y₂ + y₁ * x₂)

-- Statement: Distributive law of addition over multiplication does NOT hold
theorem distributive_law_addition_over_multiplication_not_hold :
  ∃ a b c : Pair, add a (mul b c) ≠ mul (add a b) (add a c) := by
  sorry

end distributive_law_addition_over_multiplication_not_hold_l2885_288544


namespace cary_shoe_savings_l2885_288569

def cost_of_shoes : ℕ := 120
def amount_saved : ℕ := 30
def earnings_per_lawn : ℕ := 5
def lawns_per_weekend : ℕ := 3

def weekends_needed : ℕ :=
  (cost_of_shoes - amount_saved) / (earnings_per_lawn * lawns_per_weekend)

theorem cary_shoe_savings : weekends_needed = 6 := by
  sorry

end cary_shoe_savings_l2885_288569


namespace xy_product_cardinality_l2885_288574

def X : Finset ℕ := {1, 2, 3, 4}
def Y : Finset ℕ := {5, 6, 7, 8}

theorem xy_product_cardinality :
  Finset.card ((X.product Y).image (λ (p : ℕ × ℕ) => p.1 * p.2)) = 15 := by
  sorry

end xy_product_cardinality_l2885_288574


namespace inequality_proof_l2885_288587

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end inequality_proof_l2885_288587


namespace correct_number_of_pupils_l2885_288580

/-- The number of pupils in a class where an error in one pupil's marks
    caused the class average to increase by half a mark. -/
def number_of_pupils : ℕ :=
  -- We define this as 20, which is the value we want to prove
  20

/-- The increase in one pupil's marks due to the error -/
def mark_increase : ℕ := 10

/-- The increase in the class average due to the error -/
def average_increase : ℚ := 1/2

theorem correct_number_of_pupils :
  mark_increase = (number_of_pupils : ℚ) * average_increase :=
sorry

end correct_number_of_pupils_l2885_288580


namespace shooting_is_impossible_coin_toss_is_random_triangle_angles_is_certain_l2885_288573

-- Define the types of events
inductive EventType
  | Impossible
  | Random
  | Certain

-- Define the events
def shooting_event : EventType := EventType.Impossible
def coin_toss_event : EventType := EventType.Random
def triangle_angles_event : EventType := EventType.Certain

-- Theorem statements
theorem shooting_is_impossible : shooting_event = EventType.Impossible := by sorry

theorem coin_toss_is_random : coin_toss_event = EventType.Random := by sorry

theorem triangle_angles_is_certain : triangle_angles_event = EventType.Certain := by sorry

end shooting_is_impossible_coin_toss_is_random_triangle_angles_is_certain_l2885_288573


namespace bob_candy_count_l2885_288523

/-- Bob's share of items -/
structure BobsShare where
  chewing_gums : ℕ
  chocolate_bars : ℕ
  assorted_candies : ℕ

/-- Definition of Bob's relationship and actions -/
structure BobInfo where
  is_sams_neighbor : Prop
  accompanies_sam : Prop
  share : BobsShare

/-- Theorem stating the number of candies Bob got -/
theorem bob_candy_count (bob : BobInfo) 
  (h1 : bob.is_sams_neighbor)
  (h2 : bob.accompanies_sam)
  (h3 : bob.share.chewing_gums = 15)
  (h4 : bob.share.chocolate_bars = 20)
  (h5 : bob.share.assorted_candies = 15) : 
  bob.share.assorted_candies = 15 := by
  sorry

end bob_candy_count_l2885_288523


namespace games_per_month_is_seven_l2885_288519

/-- Represents the number of baseball games in a season. -/
def games_per_season : ℕ := 14

/-- Represents the number of months in a season. -/
def months_per_season : ℕ := 2

/-- Calculates the number of baseball games played in a month. -/
def games_per_month : ℕ := games_per_season / months_per_season

/-- Theorem stating that the number of baseball games played in a month is 7. -/
theorem games_per_month_is_seven : games_per_month = 7 := by sorry

end games_per_month_is_seven_l2885_288519


namespace icosagon_diagonals_l2885_288541

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in an icosagon (20-sided polygon) is 170 -/
theorem icosagon_diagonals : num_diagonals 20 = 170 := by
  sorry

end icosagon_diagonals_l2885_288541


namespace chord_length_exists_point_P_l2885_288528

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define point F
def F : ℝ × ℝ := (-2, 0)

-- Define the line x = -4
def line_x_eq_neg_4 (x y : ℝ) : Prop := x = -4

-- Theorem 1: Length of the chord
theorem chord_length :
  ∃ (G : ℝ × ℝ), C₁ G.1 G.2 →
  ∃ (T : ℝ × ℝ), line_x_eq_neg_4 T.1 T.2 →
  (G.1 - F.1 = T.1 - G.1 ∧ G.2 - F.2 = T.2 - G.2) →
  ∃ (chord_length : ℝ), chord_length = 7 :=
sorry

-- Theorem 2: Existence of point P
theorem exists_point_P :
  ∃ (P : ℝ × ℝ), P = (4, 0) ∧
  ∀ (G : ℝ × ℝ), C₁ G.1 G.2 →
  (G.1 - P.1)^2 + (G.2 - P.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
sorry

end chord_length_exists_point_P_l2885_288528


namespace average_of_three_numbers_l2885_288520

theorem average_of_three_numbers (x : ℝ) (h1 : x = 33) : (x + 4*x + 2*x) / 3 = 77 := by
  sorry

end average_of_three_numbers_l2885_288520


namespace gain_percent_calculation_l2885_288576

/-- Proves that if the cost price of 50 articles equals the selling price of 40 articles, 
    then the gain percent is 25%. -/
theorem gain_percent_calculation (C S : ℝ) 
  (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 := by
  sorry

end gain_percent_calculation_l2885_288576


namespace min_value_expression_l2885_288566

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + 
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 
    (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + 
    (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) = 2) := by
  sorry

end min_value_expression_l2885_288566


namespace division_problem_l2885_288595

theorem division_problem : (8900 / 6) / 4 = 1483 + 1/3 := by sorry

end division_problem_l2885_288595


namespace blue_markers_count_l2885_288504

def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l2885_288504


namespace y1_greater_than_y2_l2885_288536

/-- Given two points on a linear function, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -(-1) + 1) → (y₂ = -(2) + 1) → y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l2885_288536


namespace number_puzzle_l2885_288545

theorem number_puzzle (a b : ℕ) : 
  a + b = 21875 →
  (a % 5 = 0 ∨ b % 5 = 0) →
  b = 10 * a + 5 →
  b - a = 17893 := by
sorry

end number_puzzle_l2885_288545


namespace yarn_length_multiple_l2885_288558

theorem yarn_length_multiple (green_length red_length total_length x : ℝ) : 
  green_length = 156 →
  red_length = green_length * x + 8 →
  total_length = green_length + red_length →
  total_length = 632 →
  x = 3 := by
sorry

end yarn_length_multiple_l2885_288558


namespace sin_75_times_sin_15_l2885_288577

theorem sin_75_times_sin_15 : Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by
  sorry

end sin_75_times_sin_15_l2885_288577


namespace average_time_per_km_l2885_288597

-- Define the race distance in kilometers
def race_distance : ℝ := 10

-- Define the time for the first half of the race in minutes
def first_half_time : ℝ := 20

-- Define the time for the second half of the race in minutes
def second_half_time : ℝ := 30

-- Theorem statement
theorem average_time_per_km (total_time : ℝ) (avg_time_per_km : ℝ) :
  total_time = first_half_time + second_half_time →
  avg_time_per_km = total_time / race_distance →
  avg_time_per_km = 5 := by
  sorry


end average_time_per_km_l2885_288597


namespace h_j_h_3_eq_86_l2885_288538

def h (x : ℝ) : ℝ := 2 * x + 2

def j (x : ℝ) : ℝ := 5 * x + 2

theorem h_j_h_3_eq_86 : h (j (h 3)) = 86 := by
  sorry

end h_j_h_3_eq_86_l2885_288538


namespace interest_rate_calculation_l2885_288543

/-- Given the compound interest for the second and third years, calculate the interest rate. -/
theorem interest_rate_calculation (CI2 CI3 : ℝ) (h1 : CI2 = 1200) (h2 : CI3 = 1272) :
  ∃ (r : ℝ), r = 0.06 ∧ CI3 - CI2 = CI2 * r :=
by sorry

end interest_rate_calculation_l2885_288543


namespace prob_second_red_given_first_red_l2885_288551

/-- The probability of drawing a red ball on the second draw, given that a red ball was drawn on the first -/
theorem prob_second_red_given_first_red 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (white_balls : ℕ) 
  (h1 : total_balls = 6)
  (h2 : red_balls = 4)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls - 1 : ℚ) / (total_balls - 1) = 3/5 := by
  sorry

end prob_second_red_given_first_red_l2885_288551


namespace percentage_euros_to_dollars_l2885_288592

/-- Converts a percentage of Euros to US Dollars -/
theorem percentage_euros_to_dollars
  (X : ℝ) -- Unknown amount in Euros
  (Y : ℝ) -- Exchange rate (1 Euro = Y US Dollars)
  (h : Y > 0) -- Y is positive
  : (25 / 100 : ℝ) * X * Y = 0.25 * X * Y := by
  sorry

end percentage_euros_to_dollars_l2885_288592


namespace friend_money_pooling_l2885_288513

/-- Represents the money pooling problem with 4 friends --/
theorem friend_money_pooling
  (peter john quincy andrew : ℕ)  -- Money amounts for each friend
  (h1 : peter = 320)              -- Peter has $320
  (h2 : peter = 2 * john)         -- Peter has twice as much as John
  (h3 : quincy > peter)           -- Quincy has more than Peter
  (h4 : andrew = (115 * quincy) / 100)  -- Andrew has 15% more than Quincy
  (h5 : peter + john + quincy + andrew = 1211)  -- Total money after spending $1200
  : quincy - peter = 20 :=
by sorry

end friend_money_pooling_l2885_288513


namespace boat_speed_in_still_water_l2885_288533

/-- The speed of a boat in still water, given the rate of current and distance travelled downstream. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 4 →
  downstream_distance = 10.4 →
  downstream_time = 24 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 22 ∧ downstream_distance = (boat_speed + current_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l2885_288533


namespace magic_8_ball_three_out_of_six_l2885_288507

def magic_8_ball_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem magic_8_ball_three_out_of_six :
  magic_8_ball_probability 6 3 (1/4) = 135/1024 := by
  sorry

end magic_8_ball_three_out_of_six_l2885_288507


namespace shopkeeper_cloth_sale_l2885_288593

/-- Proves the total amount a shopkeeper receives for selling cloth at a loss. -/
theorem shopkeeper_cloth_sale (total_metres : ℕ) (cost_price_per_metre : ℕ) (loss_per_metre : ℕ) : 
  total_metres = 600 →
  cost_price_per_metre = 70 →
  loss_per_metre = 10 →
  (total_metres * (cost_price_per_metre - loss_per_metre) : ℕ) = 36000 := by
  sorry

end shopkeeper_cloth_sale_l2885_288593


namespace max_female_students_with_4_teachers_min_total_people_l2885_288554

/-- Represents a study group composition --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

/-- The maximum number of female students when there are 4 teachers is 6 --/
theorem max_female_students_with_4_teachers :
  ∀ g : StudyGroup, is_valid_group g → g.teachers = 4 → g.female_students ≤ 6 := by
  sorry

/-- The minimum number of people in a valid study group is 12 --/
theorem min_total_people :
  ∀ g : StudyGroup, is_valid_group g →
    g.male_students + g.female_students + g.teachers ≥ 12 := by
  sorry

end max_female_students_with_4_teachers_min_total_people_l2885_288554


namespace max_power_of_two_divides_l2885_288572

/-- The highest power of 2 dividing a natural number -/
def v2 (n : ℕ) : ℕ := sorry

/-- The maximum power of 2 dividing (2019^n - 1) / 2018 for positive integer n -/
def max_power_of_two (n : ℕ+) : ℕ :=
  if n.val % 2 = 1 then 0 else v2 n.val + 1

/-- Theorem stating the maximum power of 2 dividing the given expression -/
theorem max_power_of_two_divides (n : ℕ+) :
  (2019^n.val - 1) / 2018 % 2^(max_power_of_two n) = 0 ∧
  ∀ k > max_power_of_two n, (2019^n.val - 1) / 2018 % 2^k ≠ 0 :=
sorry

end max_power_of_two_divides_l2885_288572


namespace addition_sequence_terms_l2885_288568

/-- Represents the nth term of the first sequence in the addition pattern -/
def a (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the nth term of the second sequence in the addition pattern -/
def b (n : ℕ) : ℕ := 5 * n - 1

/-- Proves the correctness of the 10th and 80th terms in the addition sequence -/
theorem addition_sequence_terms :
  (a 10 = 21 ∧ b 10 = 49) ∧ (a 80 = 161 ∧ b 80 = 399) := by
  sorry

#eval a 10  -- Expected: 21
#eval b 10  -- Expected: 49
#eval a 80  -- Expected: 161
#eval b 80  -- Expected: 399

end addition_sequence_terms_l2885_288568


namespace line_inclination_l2885_288575

theorem line_inclination (a : ℝ) : 
  (((2 - (-3)) / (1 - a) = Real.tan (135 * π / 180)) → a = 6) := by
  sorry

end line_inclination_l2885_288575


namespace school_gender_ratio_l2885_288581

theorem school_gender_ratio (num_girls : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  num_girls = 1200 →
  ratio_boys = 5 →
  ratio_girls = 4 →
  (ratio_boys : ℚ) / ratio_girls * num_girls = 1500 :=
by sorry

end school_gender_ratio_l2885_288581


namespace factors_of_180_l2885_288511

/-- The number of distinct positive factors of 180 -/
def num_factors_180 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 180 is 18 -/
theorem factors_of_180 : num_factors_180 = 18 := by sorry

end factors_of_180_l2885_288511


namespace flagpole_height_l2885_288553

/-- Given a person and a flagpole under the same lighting conditions, 
    we can determine the height of the flagpole using the ratio of heights to shadow lengths. -/
theorem flagpole_height
  (person_height : ℝ)
  (person_shadow : ℝ)
  (flagpole_shadow : ℝ)
  (h_person_height : person_height = 1.6)
  (h_person_shadow : person_shadow = 0.4)
  (h_flagpole_shadow : flagpole_shadow = 5)
  (h_positive : person_height > 0 ∧ person_shadow > 0 ∧ flagpole_shadow > 0) :
  (person_height / person_shadow) * flagpole_shadow = 20 :=
sorry

#check flagpole_height

end flagpole_height_l2885_288553


namespace old_supervisor_salary_proof_l2885_288550

/-- Calculates the old supervisor's salary given the initial and new average salaries,
    number of workers, and new supervisor's salary. -/
def old_supervisor_salary (initial_avg : ℚ) (new_avg : ℚ) (num_workers : ℕ) 
  (new_supervisor_salary : ℚ) : ℚ :=
  (initial_avg * (num_workers + 1) - new_avg * (num_workers + 1) + new_supervisor_salary)

/-- Proves that the old supervisor's salary was $870 given the problem conditions. -/
theorem old_supervisor_salary_proof :
  old_supervisor_salary 430 440 8 960 = 870 := by
  sorry

#eval old_supervisor_salary 430 440 8 960

end old_supervisor_salary_proof_l2885_288550


namespace blue_tile_fraction_is_three_fourths_l2885_288529

/-- Represents the tiling pattern of an 8x8 square -/
structure TilingPattern :=
  (size : Nat)
  (blue_tiles_per_corner : Nat)
  (total_corners : Nat)

/-- The fraction of blue tiles in the tiling pattern -/
def blue_tile_fraction (pattern : TilingPattern) : Rat :=
  let total_blue_tiles := pattern.blue_tiles_per_corner * pattern.total_corners
  let total_tiles := pattern.size * pattern.size
  total_blue_tiles / total_tiles

/-- Theorem stating that the fraction of blue tiles in the given pattern is 3/4 -/
theorem blue_tile_fraction_is_three_fourths (pattern : TilingPattern) 
  (h1 : pattern.size = 8)
  (h2 : pattern.blue_tiles_per_corner = 12)
  (h3 : pattern.total_corners = 4) : 
  blue_tile_fraction pattern = 3/4 := by
  sorry

#check blue_tile_fraction_is_three_fourths

end blue_tile_fraction_is_three_fourths_l2885_288529


namespace milk_storage_calculation_l2885_288514

/-- Calculates the final amount of milk in a storage tank given initial amount,
    pumping out rate and duration, and adding rate and duration. -/
def final_milk_amount (initial : ℝ) (pump_rate : ℝ) (pump_duration : ℝ) 
                       (add_rate : ℝ) (add_duration : ℝ) : ℝ :=
  initial - pump_rate * pump_duration + add_rate * add_duration

/-- Theorem stating that given the specific conditions from the problem,
    the final amount of milk in the storage tank is 28,980 gallons. -/
theorem milk_storage_calculation :
  final_milk_amount 30000 2880 4 1500 7 = 28980 := by
  sorry

end milk_storage_calculation_l2885_288514


namespace largest_angle_in_special_triangle_l2885_288516

/-- The largest possible angle in a triangle with two sides of length 2 and the third side greater than 4 --/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ) (C : ℝ),
    a = 2 →
    b = 2 →
    c > 4 →
    C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
    ∀ ε > 0, C < 180 - ε := by
  sorry

end largest_angle_in_special_triangle_l2885_288516


namespace min_shots_battleship_l2885_288557

/-- Represents a cell on the grid -/
structure Cell :=
  (row : Nat)
  (col : Nat)

/-- Represents a ship placement on the grid -/
structure Ship :=
  (start : Cell)
  (horizontal : Bool)

/-- The grid size -/
def gridSize : Nat := 5

/-- The ship length -/
def shipLength : Nat := 4

/-- Checks if a ship placement is valid on the grid -/
def isValidShip (s : Ship) : Prop :=
  s.start.row ≥ 1 ∧ s.start.row ≤ gridSize ∧
  s.start.col ≥ 1 ∧ s.start.col ≤ gridSize ∧
  (if s.horizontal
   then s.start.col + shipLength - 1 ≤ gridSize
   else s.start.row + shipLength - 1 ≤ gridSize)

/-- Checks if a shot hits a ship -/
def hitShip (shot : Cell) (s : Ship) : Prop :=
  if s.horizontal
  then shot.row = s.start.row ∧ shot.col ≥ s.start.col ∧ shot.col < s.start.col + shipLength
  else shot.col = s.start.col ∧ shot.row ≥ s.start.row ∧ shot.row < s.start.row + shipLength

/-- The main theorem: 6 shots are sufficient and necessary -/
theorem min_shots_battleship :
  ∃ (shots : Finset Cell),
    shots.card = 6 ∧
    (∀ s : Ship, isValidShip s → ∃ shot ∈ shots, hitShip shot s) ∧
    (∀ (shots' : Finset Cell), shots'.card < 6 →
      ∃ s : Ship, isValidShip s ∧ ∀ shot ∈ shots', ¬hitShip shot s) :=
sorry

end min_shots_battleship_l2885_288557


namespace prob_green_is_one_eighth_l2885_288500

-- Define the number of cubes for each color
def pink_cubes : ℕ := 36
def blue_cubes : ℕ := 18
def green_cubes : ℕ := 9
def red_cubes : ℕ := 6
def purple_cubes : ℕ := 3

-- Define the total number of cubes
def total_cubes : ℕ := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes

-- Define the probability of selecting a green cube
def prob_green : ℚ := green_cubes / total_cubes

-- Theorem statement
theorem prob_green_is_one_eighth : prob_green = 1 / 8 := by
  sorry

end prob_green_is_one_eighth_l2885_288500


namespace log_sum_equals_two_l2885_288560

theorem log_sum_equals_two : Real.log 3 / Real.log 6 + Real.log 4 / Real.log 6 = 2 := by
  sorry

end log_sum_equals_two_l2885_288560


namespace prime_sequence_equality_l2885_288531

theorem prime_sequence_equality (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p + 10)) 
  (h3 : Nat.Prime (p + 14)) 
  (h4 : Nat.Prime (2 * p + 1)) 
  (h5 : Nat.Prime (4 * p + 1)) : p = 3 := by
  sorry

end prime_sequence_equality_l2885_288531
