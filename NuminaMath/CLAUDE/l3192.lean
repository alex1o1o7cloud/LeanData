import Mathlib

namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l3192_319268

/-- Proves that mixing 300 mL of 10% alcohol solution with 450 mL of 30% alcohol solution results in a 22% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let y_volume : ℝ := 450
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.22
  
  x_volume * x_concentration + y_volume * y_concentration = 
    (x_volume + y_volume) * target_concentration :=
by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l3192_319268


namespace NUMINAMATH_CALUDE_representable_multiple_of_three_l3192_319259

/-- A number is representable if it can be written as x^2 + 2y^2 for some integers x and y -/
def Representable (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + 2*y^2

/-- If 3a is representable, then a is representable -/
theorem representable_multiple_of_three (a : ℤ) :
  Representable (3*a) → Representable a := by
  sorry

end NUMINAMATH_CALUDE_representable_multiple_of_three_l3192_319259


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3192_319241

theorem stratified_sampling_problem (high_school_students : ℕ) (middle_school_students : ℕ)
  (middle_school_sample : ℕ) (total_sample : ℕ) :
  high_school_students = 3500 →
  middle_school_students = 1500 →
  middle_school_sample = 30 →
  (middle_school_students : ℚ) / (high_school_students + middle_school_students : ℚ) * middle_school_sample = total_sample →
  total_sample = 100 := by
sorry


end NUMINAMATH_CALUDE_stratified_sampling_problem_l3192_319241


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_37_l3192_319265

theorem modular_inverse_of_5_mod_37 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 36 ∧ (5 * x) % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_37_l3192_319265


namespace NUMINAMATH_CALUDE_max_cables_cut_theorem_l3192_319284

/-- Represents a computer network -/
structure ComputerNetwork where
  num_computers : ℕ
  num_cables : ℕ
  num_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.num_cables - (network.num_computers - network.num_clusters)

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem (network : ComputerNetwork) 
  (h1 : network.num_computers = 200)
  (h2 : network.num_cables = 345)
  (h3 : network.num_clusters = 8) :
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut ⟨200, 345, 8⟩

end NUMINAMATH_CALUDE_max_cables_cut_theorem_l3192_319284


namespace NUMINAMATH_CALUDE_remainder_theorem_l3192_319235

theorem remainder_theorem (x y z : ℤ) 
  (hx : x % 102 = 56)
  (hy : y % 154 = 79)
  (hz : z % 297 = 183) :
  (x % 19 = 18) ∧ (y % 22 = 13) ∧ (z % 33 = 18) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3192_319235


namespace NUMINAMATH_CALUDE_seventh_grade_percentage_l3192_319243

theorem seventh_grade_percentage 
  (seventh_graders : ℕ) 
  (sixth_graders : ℕ) 
  (sixth_grade_percentage : ℚ) :
  seventh_graders = 64 →
  sixth_graders = 76 →
  sixth_grade_percentage = 38/100 →
  (↑seventh_graders : ℚ) / (↑sixth_graders / sixth_grade_percentage) = 32/100 :=
by sorry

end NUMINAMATH_CALUDE_seventh_grade_percentage_l3192_319243


namespace NUMINAMATH_CALUDE_intersection_point_unique_l3192_319271

/-- The line equation (x+3)/2 = (y-1)/3 = (z-1)/5 -/
def line_eq (x y z : ℝ) : Prop :=
  (x + 3) / 2 = (y - 1) / 3 ∧ (y - 1) / 3 = (z - 1) / 5

/-- The plane equation 2x + 3y + 7z - 52 = 0 -/
def plane_eq (x y z : ℝ) : Prop :=
  2 * x + 3 * y + 7 * z - 52 = 0

/-- The intersection point (-1, 4, 6) -/
def intersection_point : ℝ × ℝ × ℝ := (-1, 4, 6)

theorem intersection_point_unique :
  ∀ x y z : ℝ, line_eq x y z ∧ plane_eq x y z ↔ (x, y, z) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l3192_319271


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3192_319287

theorem complex_modulus_problem (z : ℂ) (h : (z + 2) / (z - 2) = Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3192_319287


namespace NUMINAMATH_CALUDE_min_value_of_function_l3192_319237

theorem min_value_of_function (t : ℝ) (h : t > 0) :
  (t^2 - 4*t + 1) / t ≥ -2 ∧ ∃ t > 0, (t^2 - 4*t + 1) / t = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3192_319237


namespace NUMINAMATH_CALUDE_race_heartbeats_l3192_319216

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proves that the total number of heartbeats during a 30-mile race is 21600 -/
theorem race_heartbeats :
  let heart_rate : ℕ := 120  -- beats per minute
  let pace : ℕ := 6          -- minutes per mile
  let distance : ℕ := 30     -- miles
  total_heartbeats heart_rate pace distance = 21600 := by
sorry

#eval total_heartbeats 120 6 30

end NUMINAMATH_CALUDE_race_heartbeats_l3192_319216


namespace NUMINAMATH_CALUDE_robs_double_cards_fraction_l3192_319247

theorem robs_double_cards_fraction (total_cards : ℕ) (jess_doubles : ℕ) (jess_ratio : ℕ) :
  total_cards = 24 →
  jess_doubles = 40 →
  jess_ratio = 5 →
  (jess_doubles / jess_ratio : ℚ) / total_cards = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_robs_double_cards_fraction_l3192_319247


namespace NUMINAMATH_CALUDE_shepherd_boys_sticks_l3192_319296

theorem shepherd_boys_sticks (x : ℕ) : 6 * x + 14 = 8 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_shepherd_boys_sticks_l3192_319296


namespace NUMINAMATH_CALUDE_a_outside_interval_l3192_319222

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

-- State the theorem
theorem a_outside_interval (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on_nonpositive f) 
  (h_inequality : f a > f 2) : 
  a < -2 ∨ a > 2 :=
sorry

end NUMINAMATH_CALUDE_a_outside_interval_l3192_319222


namespace NUMINAMATH_CALUDE_diamond_three_eight_l3192_319225

-- Define the operation ⋄
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_eight : diamond 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_eight_l3192_319225


namespace NUMINAMATH_CALUDE_inequality_proof_l3192_319221

theorem inequality_proof (x a : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x => x^2 - x + 1) 
  (h2 : |x - a| < 1) : 
  |f x - f a| < 2 * (|a| + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3192_319221


namespace NUMINAMATH_CALUDE_grand_forest_trail_length_l3192_319242

/-- Represents the length of Jamie's hike on the Grand Forest Trail -/
def GrandForestTrail : Type :=
  { hike : Vector ℝ 5 // 
    hike.get 0 + hike.get 1 + hike.get 2 = 42 ∧
    (hike.get 1 + hike.get 2) / 2 = 15 ∧
    hike.get 3 + hike.get 4 = 40 ∧
    hike.get 0 + hike.get 3 = 36 }

/-- The total length of the Grand Forest Trail is 82 miles -/
theorem grand_forest_trail_length (hike : GrandForestTrail) :
  hike.val.get 0 + hike.val.get 1 + hike.val.get 2 + hike.val.get 3 + hike.val.get 4 = 82 :=
by sorry

end NUMINAMATH_CALUDE_grand_forest_trail_length_l3192_319242


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l3192_319232

/-- Given two sequences (aₖ) and (bₖ) satisfying certain conditions, 
    prove that aₖ > bₖ for all k between 2 and n-1 inclusive. -/
theorem arithmetic_geometric_inequality (n : ℕ) (a b : ℕ → ℝ) 
  (h_n : n ≥ 3)
  (h_a_arith : ∀ k l : ℕ, k < l → l ≤ n → a l - a k = (l - k) * (a 2 - a 1))
  (h_b_geom : ∀ k l : ℕ, k < l → l ≤ n → b l / b k = (b 2 / b 1) ^ (l - k))
  (h_a_pos : ∀ k : ℕ, k ≤ n → 0 < a k)
  (h_b_pos : ∀ k : ℕ, k ≤ n → 0 < b k)
  (h_a_inc : ∀ k : ℕ, k < n → a k < a (k + 1))
  (h_b_inc : ∀ k : ℕ, k < n → b k < b (k + 1))
  (h_eq_first : a 1 = b 1)
  (h_eq_last : a n = b n) :
  ∀ k : ℕ, 2 ≤ k → k < n → a k > b k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l3192_319232


namespace NUMINAMATH_CALUDE_alice_profit_l3192_319249

/-- Calculates the profit from selling friendship bracelets -/
def calculate_profit (total_bracelets : ℕ) (material_cost : ℚ) (given_away : ℕ) (price_per_bracelet : ℚ) : ℚ :=
  let bracelets_sold := total_bracelets - given_away
  let revenue := (bracelets_sold : ℚ) * price_per_bracelet
  revenue - material_cost

/-- Theorem: Alice's profit from selling friendship bracelets is $8.00 -/
theorem alice_profit :
  calculate_profit 52 3 8 (1/4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_alice_profit_l3192_319249


namespace NUMINAMATH_CALUDE_equal_probabilities_l3192_319227

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box :=
  ({red := 100, green := 0}, {red := 0, green := 100})

/-- The number of balls transferred between boxes -/
def transfer_count : ℕ := 8

/-- The final state after transferring balls -/
def final_state : Box × Box :=
  let (red_box, green_box) := initial_state
  let red_box' := {red := red_box.red - transfer_count, green := transfer_count}
  let green_box' := {red := transfer_count, green := green_box.green}
  (red_box', green_box')

/-- The probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red / (box.red + box.green)
  | "green" => box.green / (box.red + box.green)
  | _ => 0

theorem equal_probabilities :
  let (final_red_box, final_green_box) := final_state
  prob_draw final_red_box "green" = prob_draw final_green_box "red" := by
  sorry


end NUMINAMATH_CALUDE_equal_probabilities_l3192_319227


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l3192_319252

/-- Sequence without consecutive ones -/
def SeqWithoutConsecutiveOnes (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => SeqWithoutConsecutiveOnes (n+1) + SeqWithoutConsecutiveOnes n

/-- Total number of possible sequences -/
def TotalSequences (n : ℕ) : ℕ := 2^n

theorem probability_no_consecutive_ones :
  (SeqWithoutConsecutiveOnes 12 : ℚ) / (TotalSequences 12) = 377 / 4096 := by
  sorry

#eval SeqWithoutConsecutiveOnes 12
#eval TotalSequences 12

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l3192_319252


namespace NUMINAMATH_CALUDE_smallest_special_number_l3192_319280

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_special_number : 
  ∀ n : ℕ, n > 0 → n % 20 = 0 → is_perfect_cube (n^2) → is_perfect_square (n^3) → 
  n ≥ 1000000 :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l3192_319280


namespace NUMINAMATH_CALUDE_cube_gt_iff_gt_l3192_319208

theorem cube_gt_iff_gt (a b : ℝ) : a^3 > b^3 ↔ a > b := by sorry

end NUMINAMATH_CALUDE_cube_gt_iff_gt_l3192_319208


namespace NUMINAMATH_CALUDE_tan_difference_special_angle_l3192_319293

theorem tan_difference_special_angle (α : Real) :
  2 * Real.tan α = 3 * Real.tan (π / 8) →
  Real.tan (α - π / 8) = (5 * Real.sqrt 2 + 1) / 49 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_special_angle_l3192_319293


namespace NUMINAMATH_CALUDE_cookies_per_pack_l3192_319291

theorem cookies_per_pack (num_trays : ℕ) (cookies_per_tray : ℕ) (num_packs : ℕ) :
  num_trays = 4 →
  cookies_per_tray = 24 →
  num_packs = 8 →
  (num_trays * cookies_per_tray) / num_packs = 12 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_pack_l3192_319291


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_5_6_l3192_319238

theorem greatest_four_digit_divisible_by_3_5_6 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 9990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_5_6_l3192_319238


namespace NUMINAMATH_CALUDE_greatest_y_value_l3192_319248

theorem greatest_y_value (y : ℝ) : 3 * y^2 + 5 * y + 3 = 3 → y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_greatest_y_value_l3192_319248


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l3192_319279

theorem greatest_integer_problem : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k l : ℕ), n = 9 * k - 2 ∧ n = 6 * l - 4) ∧
  (∀ (m : ℕ), m < 150 → 
    (∃ (k' l' : ℕ), m = 9 * k' - 2 ∧ m = 6 * l' - 4) → 
    m ≤ n) ∧
  n = 146 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l3192_319279


namespace NUMINAMATH_CALUDE_pauls_money_duration_l3192_319204

/-- Given Paul's earnings and weekly spending, prove how long his money will last. -/
theorem pauls_money_duration (lawn_mowing : ℕ) (weed_eating : ℕ) (weekly_spending : ℕ) :
  lawn_mowing = 68 →
  weed_eating = 13 →
  weekly_spending = 9 →
  (lawn_mowing + weed_eating) / weekly_spending = 9 := by
  sorry

#check pauls_money_duration

end NUMINAMATH_CALUDE_pauls_money_duration_l3192_319204


namespace NUMINAMATH_CALUDE_difference_of_squares_625_375_l3192_319257

theorem difference_of_squares_625_375 : 625^2 - 375^2 = 250000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_625_375_l3192_319257


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l3192_319258

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ 
  ¬(∀ x : ℝ, |x| > 1 → x > 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l3192_319258


namespace NUMINAMATH_CALUDE_cans_for_final_rooms_l3192_319213

-- Define the initial and final number of rooms that can be painted
def initial_rooms : ℕ := 50
def final_rooms : ℕ := 42

-- Define the number of cans lost
def cans_lost : ℕ := 4

-- Define the function to calculate the number of cans needed for a given number of rooms
def cans_needed (rooms : ℕ) : ℕ :=
  rooms / ((initial_rooms - final_rooms) / cans_lost)

-- Theorem statement
theorem cans_for_final_rooms :
  cans_needed final_rooms = 21 :=
sorry

end NUMINAMATH_CALUDE_cans_for_final_rooms_l3192_319213


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_75_by_150_percent_l3192_319281

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + (initial * percentage / 100) := by sorry

theorem increase_75_by_150_percent :
  75 * (1 + 150 / 100) = 187.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_75_by_150_percent_l3192_319281


namespace NUMINAMATH_CALUDE_solution_range_l3192_319272

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem solution_range (a b c : ℝ) :
  (f a b c 3 = 0.5) →
  (f a b c 4 = -0.5) →
  (f a b c 5 = -1) →
  ∃ x : ℝ, (ax^2 + b*x + c = 0) ∧ (3 < x) ∧ (x < 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l3192_319272


namespace NUMINAMATH_CALUDE_decreasing_interval_of_even_function_l3192_319269

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + 4

theorem decreasing_interval_of_even_function (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  {x : ℝ | ∀ y, x ≤ y → f m x ≥ f m y} = Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_even_function_l3192_319269


namespace NUMINAMATH_CALUDE_tom_apple_problem_l3192_319211

theorem tom_apple_problem (num_apples : ℕ) : 
  let total_slices := num_apples * 8
  let remaining_after_jerry := total_slices * (5/8 : ℚ)
  let remaining_after_eating := remaining_after_jerry * (1/2 : ℚ)
  remaining_after_eating = 5 →
  num_apples = 2 := by
sorry

end NUMINAMATH_CALUDE_tom_apple_problem_l3192_319211


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3192_319244

def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

def Line := {p : ℝ × ℝ | p.2 = -p.1 + 2}

theorem circle_line_intersection (r : ℝ) (hr : r > 0) :
  ∃ (A B C : ℝ × ℝ),
    A ∈ Circle r ∧ A ∈ Line ∧
    B ∈ Circle r ∧ B ∈ Line ∧
    C ∈ Circle r ∧
    C.1 = (5/4 * A.1 + 3/4 * B.1) ∧
    C.2 = (5/4 * A.2 + 3/4 * B.2) →
  r = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3192_319244


namespace NUMINAMATH_CALUDE_problem_solution_l3192_319234

theorem problem_solution :
  (∀ x : ℝ, x + 1/x = 5 → x^2 + 1/x^2 = 23) ∧
  ((5/3)^2004 * (3/5)^2003 = 5/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3192_319234


namespace NUMINAMATH_CALUDE_largest_value_l3192_319298

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 1 - 0.1)
  (hb : b = 1 - 0.01)
  (hc : c = 1 - 0.001)
  (hd : d = 1 - 0.0001)
  (he : e = 1 - 0.00001) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l3192_319298


namespace NUMINAMATH_CALUDE_integral_reciprocal_e_l3192_319262

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem integral_reciprocal_e : ∫ x in Set.Icc (1/Real.exp 1) (Real.exp 1), f x = 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_e_l3192_319262


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3192_319270

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3192_319270


namespace NUMINAMATH_CALUDE_fourteen_stones_per_bracelet_l3192_319239

/-- Given a total number of stones and a number of bracelets, 
    calculate the number of stones per bracelet. -/
def stones_per_bracelet (total_stones : ℕ) (num_bracelets : ℕ) : ℕ :=
  total_stones / num_bracelets

/-- Theorem: Given 140 stones and 10 bracelets, 
    prove that there are 14 stones per bracelet. -/
theorem fourteen_stones_per_bracelet :
  stones_per_bracelet 140 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_stones_per_bracelet_l3192_319239


namespace NUMINAMATH_CALUDE_function_properties_l3192_319295

-- Define the function f(x) = k - 1/x
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k - 1/x

-- Theorem statement
theorem function_properties (k : ℝ) :
  -- 1. Domain of f
  (∀ x : ℝ, x ≠ 0 → f k x ∈ Set.univ) ∧
  -- 2. f is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → x < y → f k x < f k y) ∧
  -- 3. If f is odd, then k = 0
  ((∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x)) → k = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3192_319295


namespace NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l3192_319223

theorem winning_candidate_vote_percentage 
  (total_members : ℕ) 
  (votes_cast : ℕ) 
  (winning_percentage : ℚ) 
  (h1 : total_members = 1600)
  (h2 : votes_cast = 525)
  (h3 : winning_percentage = 60 / 100) : 
  (((votes_cast : ℚ) * winning_percentage) / (total_members : ℚ)) * 100 = 19.6875 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l3192_319223


namespace NUMINAMATH_CALUDE_unique_k_value_l3192_319283

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 - (k + 2) * x + 6

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  (k + 2)^2 - 4 * 3 * 6 ≥ 0

-- Define the condition that 3 is a root
def three_is_root (k : ℝ) : Prop :=
  quadratic k 3 = 0

-- The main theorem
theorem unique_k_value :
  ∃! k : ℝ, has_real_roots k ∧ three_is_root k :=
sorry

end NUMINAMATH_CALUDE_unique_k_value_l3192_319283


namespace NUMINAMATH_CALUDE_fraction_equality_l3192_319236

theorem fraction_equality (n : ℝ) (h : n ≥ 2) :
  1 / (n^2 - 1) = (1/2) * (1 / (n - 1) - 1 / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3192_319236


namespace NUMINAMATH_CALUDE_new_year_cards_profit_l3192_319210

/-- The profit calculation for a store selling New Year cards -/
theorem new_year_cards_profit
  (purchase_price : ℕ)
  (total_sale : ℕ)
  (h1 : purchase_price = 21)
  (h2 : total_sale = 1457)
  (h3 : ∃ (n : ℕ) (selling_price : ℕ), n * selling_price = total_sale ∧ selling_price ≤ 2 * purchase_price) :
  ∃ (n : ℕ) (selling_price : ℕ), 
    n * selling_price = total_sale ∧ 
    selling_price ≤ 2 * purchase_price ∧
    n * (selling_price - purchase_price) = 470 :=
by sorry


end NUMINAMATH_CALUDE_new_year_cards_profit_l3192_319210


namespace NUMINAMATH_CALUDE_sum_of_alternate_angles_less_than_450_l3192_319274

-- Define a heptagon
structure Heptagon where
  vertices : Fin 7 → ℝ × ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of a heptagon being inscribed in a circle
def is_inscribed (h : Heptagon) (c : Circle) : Prop :=
  ∀ i : Fin 7, dist c.center (h.vertices i) = c.radius

-- Define the property of a point being inside a polygon
def is_inside (p : ℝ × ℝ) (h : Heptagon) : Prop :=
  sorry -- Definition of a point being inside a polygon

-- Define the angle at a vertex of the heptagon
def angle_at_vertex (h : Heptagon) (i : Fin 7) : ℝ :=
  sorry -- Definition of angle at a vertex

-- Theorem statement
theorem sum_of_alternate_angles_less_than_450 (h : Heptagon) (c : Circle) :
  is_inscribed h c → is_inside c.center h →
  angle_at_vertex h 0 + angle_at_vertex h 2 + angle_at_vertex h 4 < 450 :=
sorry

end NUMINAMATH_CALUDE_sum_of_alternate_angles_less_than_450_l3192_319274


namespace NUMINAMATH_CALUDE_heart_ten_spade_probability_l3192_319224

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of tens in a standard deck -/
def NumTens : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing a specific sequence of cards -/
def SequenceProbability (firstCardProb : ℚ) (secondCardProb : ℚ) (thirdCardProb : ℚ) : ℚ :=
  firstCardProb * secondCardProb * thirdCardProb

theorem heart_ten_spade_probability :
  let probHeartNotTen := (NumHearts - 1) / StandardDeck
  let probTenAfterHeart := NumTens / (StandardDeck - 1)
  let probSpadeAfterHeartTen := NumSpades / (StandardDeck - 2)
  let probHeartTen := 1 / StandardDeck
  let probOtherTenAfterHeartTen := (NumTens - 1) / (StandardDeck - 1)
  
  SequenceProbability probHeartNotTen probTenAfterHeart probSpadeAfterHeartTen +
  SequenceProbability probHeartTen probOtherTenAfterHeartTen probSpadeAfterHeartTen = 63 / 107800 :=
by
  sorry

end NUMINAMATH_CALUDE_heart_ten_spade_probability_l3192_319224


namespace NUMINAMATH_CALUDE_fraction_simplification_l3192_319297

theorem fraction_simplification : (3 : ℚ) / 462 + (17 : ℚ) / 42 = (95 : ℚ) / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3192_319297


namespace NUMINAMATH_CALUDE_fourth_person_height_l3192_319215

/-- Heights of four people in increasing order -/
def Heights := Fin 4 → ℝ

/-- The common difference between the heights of the first three people -/
def common_difference (h : Heights) : ℝ := h 1 - h 0

theorem fourth_person_height (h : Heights) 
  (increasing : ∀ i j, i < j → h i < h j)
  (common_diff : h 2 - h 1 = h 1 - h 0)
  (last_diff : h 3 - h 2 = 6)
  (avg_height : (h 0 + h 1 + h 2 + h 3) / 4 = 77) :
  h 3 = h 0 + 2 * (common_difference h) + 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3192_319215


namespace NUMINAMATH_CALUDE_christophers_speed_l3192_319201

/-- Given a distance of 5 miles and a time of 1.25 hours, the speed is 4 miles per hour -/
theorem christophers_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 5 → time = 1.25 → speed = distance / time → speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_christophers_speed_l3192_319201


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3192_319267

/-- Theorem: If the length of a rectangle is increased by 50% and the area remains constant, 
    then the width of the rectangle must be decreased by 33.33%. -/
theorem rectangle_dimension_change (L W A : ℝ) (h1 : A = L * W) (h2 : A > 0) (h3 : L > 0) (h4 : W > 0) :
  let new_L := 1.5 * L
  let new_W := A / new_L
  (W - new_W) / W = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3192_319267


namespace NUMINAMATH_CALUDE_exists_positive_m_for_field_l3192_319229

/-- The dimensions of a rectangular field -/
def field_length (m : ℝ) : ℝ := 4*m + 6

/-- The width of a rectangular field -/
def field_width (m : ℝ) : ℝ := 2*m - 5

/-- The area of the rectangular field -/
def field_area : ℝ := 159

/-- Theorem stating that there exists a positive real number m that satisfies the field dimensions and area -/
theorem exists_positive_m_for_field : ∃ m : ℝ, m > 0 ∧ field_length m * field_width m = field_area := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_m_for_field_l3192_319229


namespace NUMINAMATH_CALUDE_hiram_allyson_age_problem_l3192_319228

/-- The number added to Hiram's age -/
def x : ℕ := 12

theorem hiram_allyson_age_problem :
  let hiram_age : ℕ := 40
  let allyson_age : ℕ := 28
  hiram_age + x = 2 * allyson_age - 4 :=
by sorry

end NUMINAMATH_CALUDE_hiram_allyson_age_problem_l3192_319228


namespace NUMINAMATH_CALUDE_school_contribution_l3192_319218

def book_cost : ℕ := 12
def num_students : ℕ := 30
def sally_paid : ℕ := 40

theorem school_contribution : 
  ∃ (school_amount : ℕ), 
    school_amount = book_cost * num_students - sally_paid ∧ 
    school_amount = 320 := by
  sorry

end NUMINAMATH_CALUDE_school_contribution_l3192_319218


namespace NUMINAMATH_CALUDE_saucer_surface_area_l3192_319254

/-- The surface area of a saucer with given dimensions -/
theorem saucer_surface_area (radius : ℝ) (rim_thickness : ℝ) (cap_height : ℝ) 
  (h1 : radius = 3)
  (h2 : rim_thickness = 1)
  (h3 : cap_height = 1.5) :
  2 * Real.pi * radius * cap_height + Real.pi * (radius^2 - (radius - rim_thickness)^2) = 14 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_saucer_surface_area_l3192_319254


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l3192_319256

theorem polynomial_coefficient_sums (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 129) ∧
  (a₁ + a₃ + a₅ + a₇ = 8256) ∧
  (a₀ + a₂ + a₄ + a₆ = -8128) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 16384) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l3192_319256


namespace NUMINAMATH_CALUDE_equation_solution_l3192_319214

theorem equation_solution : 
  ∃ x₁ x₂ : ℚ, x₁ = 8/3 ∧ x₂ = 2 ∧ 
  (∀ x : ℚ, x^2 - 6*x + 9 = (5 - 2*x)^2 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3192_319214


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l3192_319212

theorem cab_driver_average_income 
  (incomes : List ℝ) 
  (h_incomes : incomes = [400, 250, 650, 400, 500]) 
  (h_days : incomes.length = 5) : 
  (incomes.sum / incomes.length : ℝ) = 440 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l3192_319212


namespace NUMINAMATH_CALUDE_bisected_line_segment_l3192_319260

/-- Given a line segment with endpoints (5,1) and (m,1) bisected by x-2y=0, m = -1 -/
theorem bisected_line_segment (m : ℝ) : 
  let endpoint1 : ℝ × ℝ := (5, 1)
  let endpoint2 : ℝ × ℝ := (m, 1)
  let bisector : ℝ → ℝ := fun x => x / 2
  (bisector (endpoint1.1 + endpoint2.1) - 2 * 1 = 0) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_bisected_line_segment_l3192_319260


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l3192_319263

/-- The common factor of the polynomial 3ma^2 - 6mab is 3ma -/
theorem common_factor_of_polynomial (m a b : ℤ) :
  ∃ (k₁ k₂ : ℤ), 3 * m * a^2 - 6 * m * a * b = 3 * m * a * (k₁ * a + k₂ * b) :=
sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l3192_319263


namespace NUMINAMATH_CALUDE_roulette_probability_l3192_319250

/-- Represents a roulette wheel with sections A, B, and C. -/
structure RouletteWheel where
  probA : ℚ
  probB : ℚ
  probC : ℚ

/-- The sum of probabilities for all sections in a roulette wheel is 1. -/
def validWheel (wheel : RouletteWheel) : Prop :=
  wheel.probA + wheel.probB + wheel.probC = 1

/-- Theorem: Given a valid roulette wheel with probA = 1/4 and probB = 1/2, probC must be 1/4. -/
theorem roulette_probability (wheel : RouletteWheel) 
  (h_valid : validWheel wheel) 
  (h_probA : wheel.probA = 1/4) 
  (h_probB : wheel.probB = 1/2) : 
  wheel.probC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_roulette_probability_l3192_319250


namespace NUMINAMATH_CALUDE_average_salary_non_technicians_l3192_319231

/-- Proves that the average salary of non-technician workers is 6000 given the conditions --/
theorem average_salary_non_technicians (total_workers : ℕ) (avg_salary_all : ℕ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℕ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (total_workers - num_technicians) * 
    ((total_workers * avg_salary_all - num_technicians * avg_salary_technicians) / 
     (total_workers - num_technicians)) = 6000 * (total_workers - num_technicians) :=
by
  sorry

#check average_salary_non_technicians

end NUMINAMATH_CALUDE_average_salary_non_technicians_l3192_319231


namespace NUMINAMATH_CALUDE_find_C_value_l3192_319266

theorem find_C_value (D : ℝ) (h1 : 4 * C - 2 * D - 3 = 26) (h2 : D = 3) : C = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_find_C_value_l3192_319266


namespace NUMINAMATH_CALUDE_base7_difference_to_decimal_l3192_319261

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The difference between two base 7 numbers --/
def base7Difference (a b : List Nat) : List Nat :=
  sorry -- Implementation of base 7 subtraction

theorem base7_difference_to_decimal : 
  let a := [4, 1, 2, 3] -- 3214 in base 7 (least significant digit first)
  let b := [4, 3, 2, 1] -- 1234 in base 7 (least significant digit first)
  base7ToDecimal (base7Difference a b) = 721 := by
  sorry

end NUMINAMATH_CALUDE_base7_difference_to_decimal_l3192_319261


namespace NUMINAMATH_CALUDE_masha_can_pay_with_five_ruble_coins_l3192_319278

theorem masha_can_pay_with_five_ruble_coins 
  (p c n : ℕ+) 
  (h : 2 * p.val + c.val + 7 * n.val = 100) : 
  5 ∣ (p.val + 3 * c.val + n.val) := by
  sorry

end NUMINAMATH_CALUDE_masha_can_pay_with_five_ruble_coins_l3192_319278


namespace NUMINAMATH_CALUDE_curve_range_l3192_319276

/-- The curve y^2 - xy + 2x + k = 0 passes through the point (a, -a) -/
def passes_through (k a : ℝ) : Prop :=
  (-a)^2 - a * (-a) + 2 * a + k = 0

/-- The range of k values for which the curve passes through (a, -a) for some real a -/
def k_range (k : ℝ) : Prop :=
  ∃ a : ℝ, passes_through k a

theorem curve_range :
  ∀ k : ℝ, k_range k → k ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_curve_range_l3192_319276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l3192_319286

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem arithmetic_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_special : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 → 1 / m + 4 / n ≥ 3 / 2) ∧
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 ∧ 1 / m + 4 / n = 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l3192_319286


namespace NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l3192_319217

/-- The number of erasers Jungkook has -/
def jungkook_erasers : ℕ := 6

/-- The number of erasers Jimin has -/
def jimin_erasers : ℕ := jungkook_erasers + 4

/-- The number of erasers Seokjin has -/
def seokjin_erasers : ℕ := jimin_erasers - 3

theorem jungkook_has_fewest_erasers :
  jungkook_erasers < jimin_erasers ∧ jungkook_erasers < seokjin_erasers :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l3192_319217


namespace NUMINAMATH_CALUDE_base_8_4531_equals_2393_l3192_319294

def base_8_to_10 (a b c d : ℕ) : ℕ :=
  a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0

theorem base_8_4531_equals_2393 :
  base_8_to_10 4 5 3 1 = 2393 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4531_equals_2393_l3192_319294


namespace NUMINAMATH_CALUDE_parallel_segment_length_l3192_319200

/-- Represents a trapezoid with given base lengths -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  h : shorter_base > 0
  k : longer_base > shorter_base

/-- Represents a line segment parallel to the bases of a trapezoid -/
structure ParallelSegment (T : Trapezoid) where
  length : ℝ
  passes_through_diagonal_intersection : Bool

/-- The theorem statement -/
theorem parallel_segment_length 
  (T : Trapezoid) 
  (S : ParallelSegment T) 
  (h : T.shorter_base = 4) 
  (k : T.longer_base = 12) 
  (m : S.passes_through_diagonal_intersection = true) : 
  S.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l3192_319200


namespace NUMINAMATH_CALUDE_wendy_ribbon_left_l3192_319251

/-- The amount of ribbon Wendy has left after using some for wrapping presents -/
def ribbon_left (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem: Given Wendy bought 84 inches of ribbon and used 46 inches, 
    the amount of ribbon left is 38 inches -/
theorem wendy_ribbon_left : 
  ribbon_left 84 46 = 38 := by
  sorry

end NUMINAMATH_CALUDE_wendy_ribbon_left_l3192_319251


namespace NUMINAMATH_CALUDE_shortest_segment_right_triangle_l3192_319245

theorem shortest_segment_right_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  ∃ (t : ℝ), t = 2 * Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x * y = (a * b) / 2 → 
  t ≤ Real.sqrt (x^2 + y^2 - 2 * x * y * (b / c)) := by
  sorry

end NUMINAMATH_CALUDE_shortest_segment_right_triangle_l3192_319245


namespace NUMINAMATH_CALUDE_decimal_division_l3192_319230

theorem decimal_division (x y : ℚ) (hx : x = 0.25) (hy : y = 0.005) : x / y = 50 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l3192_319230


namespace NUMINAMATH_CALUDE_sine_graph_transformation_l3192_319209

theorem sine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (x + π / 6)
  let g (x : ℝ) := f (x + π / 4)
  let h (x : ℝ) := g (x / 2)
  h x = Real.sin (x / 2 + 5 * π / 12) := by sorry

end NUMINAMATH_CALUDE_sine_graph_transformation_l3192_319209


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3192_319205

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3192_319205


namespace NUMINAMATH_CALUDE_complex_exponentiation_l3192_319277

theorem complex_exponentiation (i : ℂ) (h : i * i = -1) : 
  (1 + i) ^ (2 * i) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponentiation_l3192_319277


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3192_319275

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y, x > y ∧ y > 0 → x / y > 1) ∧
  (∃ x y, x / y > 1 ∧ ¬(x > y ∧ y > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3192_319275


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3192_319219

/-- Represents a rectangular field with a specific ratio of width to length and a given perimeter. -/
structure RectangularField where
  width : ℝ
  length : ℝ
  width_length_ratio : width = length / 3
  perimeter : width * 2 + length * 2 = 72

/-- Calculates the area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem stating that a rectangular field with the given properties has an area of 243 square meters. -/
theorem rectangular_field_area (field : RectangularField) : area field = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3192_319219


namespace NUMINAMATH_CALUDE_petya_final_amount_l3192_319220

/-- Represents the juice distribution problem between Petya and Masha -/
structure JuiceDistribution where
  total : ℝ
  petya_initial : ℝ
  masha_initial : ℝ
  transferred : ℝ
  h_total : total = 10
  h_initial_sum : petya_initial + masha_initial = total
  h_after_transfer : petya_initial + transferred = 3 * (masha_initial - transferred)
  h_masha_reduction : masha_initial - transferred = (1/3) * masha_initial

/-- Theorem stating that Petya's final amount of juice is 7.5 liters -/
theorem petya_final_amount (jd : JuiceDistribution) : 
  jd.petya_initial + jd.transferred = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_petya_final_amount_l3192_319220


namespace NUMINAMATH_CALUDE_fermat_min_l3192_319273

theorem fermat_min (n : ℕ) (x y z : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n := by
  sorry

end NUMINAMATH_CALUDE_fermat_min_l3192_319273


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3192_319290

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x₀ > 0, p x₀) ↔ ∀ x > 0, ¬(p x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x₀ > 0, 2^x₀ ≥ 3) ↔ ∀ x > 0, 2^x < 3 := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3192_319290


namespace NUMINAMATH_CALUDE_distribution_difference_l3192_319292

theorem distribution_difference (total : ℕ) (p q r s : ℕ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  q = r →
  p + q + r + s = total →
  s - p = 250 := by
sorry

end NUMINAMATH_CALUDE_distribution_difference_l3192_319292


namespace NUMINAMATH_CALUDE_magic_square_sum_l3192_319240

/-- Represents a 3x3 magic square -/
def MagicSquare := Fin 3 → Fin 3 → ℕ

/-- The magic sum of a magic square -/
def magicSum (s : MagicSquare) : ℕ := s 0 0 + s 0 1 + s 0 2

/-- Predicate to check if a square is magic -/
def isMagic (s : MagicSquare) : Prop :=
  let sum := magicSum s
  (∀ i, s i 0 + s i 1 + s i 2 = sum) ∧
  (∀ j, s 0 j + s 1 j + s 2 j = sum) ∧
  (s 0 0 + s 1 1 + s 2 2 = sum) ∧
  (s 0 2 + s 1 1 + s 2 0 = sum)

theorem magic_square_sum (s : MagicSquare) (x y : ℕ) 
  (h1 : s 0 0 = x)
  (h2 : s 0 1 = 6)
  (h3 : s 0 2 = 20)
  (h4 : s 1 0 = 22)
  (h5 : s 1 1 = y)
  (h6 : isMagic s) :
  x + y = 12 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_sum_l3192_319240


namespace NUMINAMATH_CALUDE_yellow_crane_tower_visitor_l3192_319289

structure Person :=
  (name : String)
  (visited : Bool)
  (statement : Bool)

def A : Person := { name := "A", visited := false, statement := false }
def B : Person := { name := "B", visited := false, statement := false }
def C : Person := { name := "C", visited := false, statement := false }

def people : List Person := [A, B, C]

theorem yellow_crane_tower_visitor :
  (∃! p : Person, p.visited = true) →
  (∃! p : Person, p.statement = false) →
  (A.statement = (¬C.visited)) →
  (B.statement = B.visited) →
  (C.statement = A.statement) →
  A.visited = true :=
by sorry

end NUMINAMATH_CALUDE_yellow_crane_tower_visitor_l3192_319289


namespace NUMINAMATH_CALUDE_cheryl_pesto_production_l3192_319202

/-- The number of cups of basil needed to make one cup of pesto -/
def basil_per_pesto : ℕ := 4

/-- The number of cups of basil Cheryl can harvest per week -/
def basil_per_week : ℕ := 16

/-- The number of weeks Cheryl can harvest basil -/
def harvest_weeks : ℕ := 8

/-- The total number of cups of pesto Cheryl can make -/
def total_pesto : ℕ := (basil_per_week * harvest_weeks) / basil_per_pesto

theorem cheryl_pesto_production :
  total_pesto = 32 := by sorry

end NUMINAMATH_CALUDE_cheryl_pesto_production_l3192_319202


namespace NUMINAMATH_CALUDE_sixteen_bananas_equal_nineteen_grapes_l3192_319203

/-- The cost relationship between bananas, oranges, and grapes -/
structure FruitCosts where
  banana_orange_ratio : ℚ  -- 4 bananas = 3 oranges
  orange_grape_ratio : ℚ   -- 5 oranges = 8 grapes

/-- Calculate the number of grapes equivalent in cost to a given number of bananas -/
def grapes_for_bananas (costs : FruitCosts) (num_bananas : ℕ) : ℕ :=
  let oranges : ℚ := (num_bananas : ℚ) * costs.banana_orange_ratio
  let grapes : ℚ := oranges * costs.orange_grape_ratio
  grapes.ceil.toNat

/-- Theorem stating that 16 bananas cost as much as 19 grapes -/
theorem sixteen_bananas_equal_nineteen_grapes (costs : FruitCosts) 
    (h1 : costs.banana_orange_ratio = 3/4)
    (h2 : costs.orange_grape_ratio = 8/5) : 
  grapes_for_bananas costs 16 = 19 := by
  sorry

#eval grapes_for_bananas ⟨3/4, 8/5⟩ 16

end NUMINAMATH_CALUDE_sixteen_bananas_equal_nineteen_grapes_l3192_319203


namespace NUMINAMATH_CALUDE_cow_distribution_theorem_l3192_319288

/-- Represents the distribution of cows among four sons -/
structure CowDistribution where
  total : ℕ
  first_son : ℚ
  second_son : ℚ
  third_son : ℚ
  fourth_son : ℕ

/-- Theorem stating the total number of cows given the distribution -/
theorem cow_distribution_theorem (d : CowDistribution) :
  d.first_son = 1/3 ∧ 
  d.second_son = 1/5 ∧ 
  d.third_son = 1/6 ∧ 
  d.fourth_son = 12 ∧
  d.first_son + d.second_son + d.third_son + (d.fourth_son : ℚ) / d.total = 1 →
  d.total = 40 := by
  sorry

end NUMINAMATH_CALUDE_cow_distribution_theorem_l3192_319288


namespace NUMINAMATH_CALUDE_retailer_markup_percentage_l3192_319264

/-- Proves that a retailer who marks up goods by x%, offers a 15% discount, 
    and makes 27.5% profit, must have marked up the goods by 50% --/
theorem retailer_markup_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (actual_profit_percentage : ℝ)
  (h1 : discount_percentage = 15)
  (h2 : actual_profit_percentage = 27.5)
  (h3 : cost_price > 0)
  (h4 : markup_percentage > 0)
  : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + actual_profit_percentage / 100) →
  markup_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_retailer_markup_percentage_l3192_319264


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3192_319207

theorem rectangle_circle_area_ratio (w l r : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 2 * Real.pi * r) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3192_319207


namespace NUMINAMATH_CALUDE_rectangle_area_integer_l3192_319285

theorem rectangle_area_integer (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (n : ℕ), (a + b) * Real.sqrt (a * b) = n) ↔ (a = 9 ∧ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_integer_l3192_319285


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l3192_319299

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x, x^3 - 3*x^2 + 4*x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x, x^3 - 12*x^2 + 49*x - 67 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l3192_319299


namespace NUMINAMATH_CALUDE_remainder_11_power_603_mod_500_l3192_319253

theorem remainder_11_power_603_mod_500 : 11^603 % 500 = 331 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_603_mod_500_l3192_319253


namespace NUMINAMATH_CALUDE_jake_bitcoin_proportion_l3192_319206

/-- The proportion of bitcoins Jake gave to his brother -/
def proportion_to_brother : ℚ := 1/2

/-- Jake's initial fortune in bitcoins -/
def initial_fortune : ℕ := 80

/-- First donation amount in bitcoins -/
def first_donation : ℕ := 20

/-- Second donation amount in bitcoins -/
def second_donation : ℕ := 10

/-- Jake's final amount of bitcoins -/
def final_amount : ℕ := 80

theorem jake_bitcoin_proportion :
  let remaining_after_first_donation := initial_fortune - first_donation
  let remaining_after_giving_to_brother := remaining_after_first_donation * (1 - proportion_to_brother)
  let amount_after_tripling := remaining_after_giving_to_brother * 3
  amount_after_tripling - second_donation = final_amount :=
by sorry

end NUMINAMATH_CALUDE_jake_bitcoin_proportion_l3192_319206


namespace NUMINAMATH_CALUDE_sum_f_negative_l3192_319246

-- Define the function f
variable (f : ℝ → ℝ)

-- State the properties of f
axiom f_symmetry (x : ℝ) : f (4 - x) = -f x
axiom f_monotone_increasing (x y : ℝ) : x > 2 → y > x → f y > f x

-- Define the theorem
theorem sum_f_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_f_negative_l3192_319246


namespace NUMINAMATH_CALUDE_max_profit_transport_plan_l3192_319255

/-- Represents the transportation problem for fruits A, B, and C. -/
structure FruitTransport where
  total_trucks : ℕ
  total_tons : ℕ
  tons_per_truck_A : ℕ
  tons_per_truck_B : ℕ
  tons_per_truck_C : ℕ
  profit_per_ton_A : ℕ
  profit_per_ton_B : ℕ
  profit_per_ton_C : ℕ
  min_trucks_per_fruit : ℕ

/-- Calculates the profit for a given transportation plan. -/
def calculate_profit (ft : FruitTransport) (x y : ℕ) : ℕ :=
  ft.profit_per_ton_A * ft.tons_per_truck_A * x +
  ft.profit_per_ton_B * ft.tons_per_truck_B * y +
  ft.profit_per_ton_C * ft.tons_per_truck_C * (ft.total_trucks - x - y)

/-- States that the given transportation plan maximizes profit. -/
theorem max_profit_transport_plan (ft : FruitTransport)
  (h_total_trucks : ft.total_trucks = 20)
  (h_total_tons : ft.total_tons = 100)
  (h_tons_A : ft.tons_per_truck_A = 6)
  (h_tons_B : ft.tons_per_truck_B = 5)
  (h_tons_C : ft.tons_per_truck_C = 4)
  (h_profit_A : ft.profit_per_ton_A = 500)
  (h_profit_B : ft.profit_per_ton_B = 600)
  (h_profit_C : ft.profit_per_ton_C = 400)
  (h_min_trucks : ft.min_trucks_per_fruit = 2) :
  ∃ (x y : ℕ),
    x = 2 ∧
    y = 16 ∧
    ft.total_trucks - x - y = 2 ∧
    calculate_profit ft x y = 57200 ∧
    ∀ (x' y' : ℕ),
      x' ≥ ft.min_trucks_per_fruit →
      y' ≥ ft.min_trucks_per_fruit →
      ft.total_trucks - x' - y' ≥ ft.min_trucks_per_fruit →
      calculate_profit ft x' y' ≤ calculate_profit ft x y :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_transport_plan_l3192_319255


namespace NUMINAMATH_CALUDE_number_of_children_l3192_319282

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  total_pencils / pencils_per_child = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l3192_319282


namespace NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l3192_319233

theorem range_of_a_when_proposition_false (a : ℝ) :
  (∀ t : ℝ, t^2 - 2*t - a ≥ 0) → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l3192_319233


namespace NUMINAMATH_CALUDE_jessies_weight_calculation_l3192_319226

/-- Calculates Jessie's current weight after changes due to jogging, diet, and strength training -/
def jessies_current_weight (initial_weight weight_lost_jogging weight_lost_diet weight_gained_training : ℕ) : ℕ :=
  initial_weight - weight_lost_jogging - weight_lost_diet + weight_gained_training

/-- Theorem stating that Jessie's current weight is 29 kilograms -/
theorem jessies_weight_calculation :
  jessies_current_weight 69 35 10 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_jessies_weight_calculation_l3192_319226
