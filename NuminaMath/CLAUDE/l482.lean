import Mathlib

namespace sum_of_odd_coefficients_l482_48243

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (3*x - 2)^6 = a₀ + a₁*(2*x - 1) + a₂*(2*x - 1)^2 + a₃*(2*x - 1)^3 + 
                      a₄*(2*x - 1)^4 + a₅*(2*x - 1)^5 + a₆*(2*x - 1)^6) →
  a₁ + a₃ + a₅ = -63/2 := by
sorry

end sum_of_odd_coefficients_l482_48243


namespace eden_initial_bears_count_l482_48298

/-- Represents the number of stuffed bears Eden had initially --/
def eden_initial_bears : ℕ := 10

/-- Represents the total number of stuffed bears Daragh had initially --/
def daragh_initial_bears : ℕ := 20

/-- Represents the number of favorite bears Daragh took out --/
def daragh_favorite_bears : ℕ := 8

/-- Represents the number of Daragh's sisters --/
def number_of_sisters : ℕ := 3

/-- Represents the number of stuffed bears Eden has now --/
def eden_current_bears : ℕ := 14

theorem eden_initial_bears_count :
  eden_initial_bears =
    eden_current_bears -
    ((daragh_initial_bears - daragh_favorite_bears) / number_of_sisters) :=
by
  sorry

#eval eden_initial_bears

end eden_initial_bears_count_l482_48298


namespace celebrity_baby_picture_matching_probability_l482_48205

theorem celebrity_baby_picture_matching_probability :
  ∀ (n : ℕ), n = 5 →
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 120 :=
by sorry

end celebrity_baby_picture_matching_probability_l482_48205


namespace hyunji_pencils_l482_48232

/-- Given an initial number of pencils, the number given away, and the number received,
    calculate the final number of pencils. -/
def final_pencils (initial given_away received : ℕ) : ℕ :=
  initial - given_away + received

/-- Theorem stating that with 20 initial pencils, giving away 7 and receiving 5
    results in 18 pencils. -/
theorem hyunji_pencils : final_pencils 20 7 5 = 18 := by
  sorry

end hyunji_pencils_l482_48232


namespace right_triangle_third_side_l482_48220

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 6 ∧ b = 8 ∧ c > 0 ∧ a^2 + b^2 = c^2 → c = 10 :=
by sorry

end right_triangle_third_side_l482_48220


namespace bargain_bin_book_count_l482_48246

/-- Calculates the final number of books in a bargain bin after selling and adding books. -/
def final_book_count (initial : ℕ) (sold : ℕ) (added : ℕ) : ℕ :=
  initial - sold + added

/-- Proves that the final number of books in the bin is correct for the given scenario. -/
theorem bargain_bin_book_count :
  final_book_count 4 3 10 = 11 := by
  sorry

end bargain_bin_book_count_l482_48246


namespace pentagon_rectangle_ratio_l482_48248

theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side : ℝ) (rect_width : ℝ),
    pentagon_side * 5 = 60 →
    rect_width * 6 = 40 →
    pentagon_side / rect_width = 9 / 5 := by
sorry

end pentagon_rectangle_ratio_l482_48248


namespace independence_implies_a_minus_b_eq_neg_two_l482_48216

theorem independence_implies_a_minus_b_eq_neg_two :
  ∀ (a b : ℝ), 
  (∀ x : ℝ, ∃ c : ℝ, ∀ y : ℝ, x^2 + a*x - (b*y^2 - y - 3) = c) →
  a - b = -2 :=
by sorry

end independence_implies_a_minus_b_eq_neg_two_l482_48216


namespace art_museum_pictures_l482_48262

theorem art_museum_pictures : ∃ (P : ℕ), P > 0 ∧ P % 2 = 1 ∧ (P + 1) % 2 = 0 ∧ ∀ (Q : ℕ), (Q > 0 ∧ Q % 2 = 1 ∧ (Q + 1) % 2 = 0) → P ≤ Q :=
by sorry

end art_museum_pictures_l482_48262


namespace sum_of_fraction_parts_l482_48263

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.45454545

/-- Expresses the repeating decimal as a fraction -/
def as_fraction (x : ℚ) : ℚ := (100 * x - x) / 99

/-- Reduces a fraction to its lowest terms -/
def reduce_fraction (x : ℚ) : ℚ := x

theorem sum_of_fraction_parts : 
  (reduce_fraction (as_fraction repeating_decimal)).num +
  (reduce_fraction (as_fraction repeating_decimal)).den = 16 := by
  sorry

end sum_of_fraction_parts_l482_48263


namespace johann_manipulation_l482_48256

theorem johann_manipulation (x y k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hk : k > 1) : 
  x * k - y / k > x - y := by
  sorry

end johann_manipulation_l482_48256


namespace parabola_area_and_binomial_expansion_l482_48207

/-- Given a > 0 and the area enclosed by y² = ax and x = 1 is 4/3, 
    the coefficient of x⁻¹⁸ in the expansion of (x + a/x)²⁰ is 20 -/
theorem parabola_area_and_binomial_expansion (a : ℝ) (h1 : a > 0) 
  (h2 : (2 : ℝ) * ∫ x in (0 : ℝ)..(1 : ℝ), (a * x).sqrt = 4/3) :
  (Finset.range 21).sum (fun k => Nat.choose 20 k * a^k * (-1)^(19 - k)) = 20 := by
  sorry

end parabola_area_and_binomial_expansion_l482_48207


namespace simplify_and_evaluate_l482_48286

theorem simplify_and_evaluate (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) = -2 - x := by
  sorry

end simplify_and_evaluate_l482_48286


namespace sphere_volume_surface_area_ratio_l482_48212

theorem sphere_volume_surface_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 →
  (4/3 * π * r₁^3) / (4/3 * π * r₂^3) = 8 →
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 :=
by
  sorry

end sphere_volume_surface_area_ratio_l482_48212


namespace total_amount_spent_l482_48275

theorem total_amount_spent (num_pens num_pencils : ℕ) 
                           (avg_pen_price avg_pencil_price : ℚ) : 
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 14 →
  avg_pencil_price = 2 →
  (num_pens : ℚ) * avg_pen_price + (num_pencils : ℚ) * avg_pencil_price = 570 :=
by
  sorry

end total_amount_spent_l482_48275


namespace fraction_simplification_l482_48214

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end fraction_simplification_l482_48214


namespace reciprocal_in_fourth_quadrant_l482_48247

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Define the fourth quadrant
def fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

-- Theorem statement
theorem reciprocal_in_fourth_quadrant :
  fourth_quadrant (z⁻¹) := by
  sorry

end reciprocal_in_fourth_quadrant_l482_48247


namespace base5_412_to_base7_l482_48260

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 7) ((m % 7) :: acc)
    go n []

theorem base5_412_to_base7 :
  decimalToBase7 (base5ToDecimal [2, 1, 4]) = [2, 1, 2] :=
sorry

end base5_412_to_base7_l482_48260


namespace fib_sum_squares_l482_48291

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: Sum of squares of consecutive Fibonacci numbers -/
theorem fib_sum_squares (n : ℕ) : (fib n)^2 + (fib (n + 1))^2 = fib (2 * n + 2) := by
  sorry

end fib_sum_squares_l482_48291


namespace sum_due_proof_l482_48228

/-- Represents the relationship between banker's discount, true discount, and face value -/
def discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + (td * bd / fv)

/-- Proves that given a banker's discount of 36 and a true discount of 30,
    the face value (sum due) is 180 -/
theorem sum_due_proof :
  ∃ (fv : ℚ), discount_relation 36 30 fv ∧ fv = 180 := by
  sorry

end sum_due_proof_l482_48228


namespace uncovered_side_length_l482_48215

/-- Proves that for a rectangular field with given area and fencing length, 
    the length of the uncovered side is as specified. -/
theorem uncovered_side_length 
  (area : ℝ) 
  (fencing_length : ℝ) 
  (h_area : area = 680) 
  (h_fencing : fencing_length = 178) : 
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * width + length = fencing_length ∧ 
    length = 170 := by
  sorry

end uncovered_side_length_l482_48215


namespace tetrahedron_volume_in_cube_l482_48239

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) 
  (h_side_length : cube_side_length = 10) : ℝ :=
by
  -- The volume of the tetrahedron formed by alternately colored vertices
  -- of a cube with side length 10 units is 1000/3 cubic units
  sorry

#check tetrahedron_volume_in_cube

end tetrahedron_volume_in_cube_l482_48239


namespace product_equals_three_halves_l482_48290

theorem product_equals_three_halves : 12 * 0.5 * 4 * 0.0625 = 3/2 := by
  sorry

end product_equals_three_halves_l482_48290


namespace simplify_fraction_l482_48209

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) := by
  sorry

end simplify_fraction_l482_48209


namespace cookie_jar_problem_l482_48271

theorem cookie_jar_problem (x : ℕ) : 
  (x - 1 = (x + 5) / 2) → x = 7 := by
  sorry

end cookie_jar_problem_l482_48271


namespace problem_solution_l482_48230

def f (a : ℝ) (x : ℝ) : ℝ := |x| * (x - a)

theorem problem_solution :
  (∀ x, f a x = -f a (-x)) → a = 0 ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f a x ≤ f a y) → a ≤ 0 ∧
  ∃ a, a < 0 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1/2 → f a x ≤ 2) ∧
     (∃ x, -1 ≤ x ∧ x ≤ 1/2 ∧ f a x = 2) ∧
     a = -3 :=
by sorry

end problem_solution_l482_48230


namespace seven_strip_trapezoid_shaded_area_l482_48219

/-- Represents a trapezoid divided into equal width strips -/
structure StripTrapezoid where
  numStrips : ℕ
  numShaded : ℕ
  h_pos : 0 < numStrips
  h_shaded : numShaded ≤ numStrips

/-- The fraction of shaded area in a strip trapezoid -/
def shadedAreaFraction (t : StripTrapezoid) : ℚ :=
  t.numShaded / t.numStrips

/-- Theorem: In a trapezoid divided into 7 strips with 4 shaded, the shaded area is 4/7 of the total area -/
theorem seven_strip_trapezoid_shaded_area :
  let t : StripTrapezoid := ⟨7, 4, by norm_num, by norm_num⟩
  shadedAreaFraction t = 4 / 7 := by
  sorry

end seven_strip_trapezoid_shaded_area_l482_48219


namespace count_ordered_pairs_l482_48208

theorem count_ordered_pairs : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 * p.2 = 20^20) (Finset.product (Finset.range (20^20 + 1)) (Finset.range (20^20 + 1)))).card = 231 :=
sorry

end count_ordered_pairs_l482_48208


namespace average_of_s_and_t_l482_48252

theorem average_of_s_and_t (s t : ℝ) : 
  (1 + 3 + 7 + s + t) / 5 = 12 → (s + t) / 2 = 24.5 := by
sorry

end average_of_s_and_t_l482_48252


namespace hike_length_is_four_l482_48294

/-- Represents the hike details -/
structure Hike where
  initial_water : ℝ
  duration : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_part_consumption_rate : ℝ

/-- Calculates the length of the hike in miles -/
def hike_length (h : Hike) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, the hike length is 4 miles -/
theorem hike_length_is_four (h : Hike) 
  (h_initial : h.initial_water = 10)
  (h_duration : h.duration = 2)
  (h_remaining : h.remaining_water = 2)
  (h_leak : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 3)
  (h_first_part : h.first_part_consumption_rate = 1) :
  hike_length h = 4 := by
  sorry

end hike_length_is_four_l482_48294


namespace a_squared_gt_a_necessary_not_sufficient_l482_48257

theorem a_squared_gt_a_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ ¬(a > 1)) :=
by sorry

end a_squared_gt_a_necessary_not_sufficient_l482_48257


namespace square_fencing_cost_l482_48254

/-- The cost of fencing one side of a square -/
def cost_per_side : ℕ := 69

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing a square -/
def total_cost : ℕ := cost_per_side * num_sides

theorem square_fencing_cost : total_cost = 276 := by
  sorry

end square_fencing_cost_l482_48254


namespace right_triangle_median_length_l482_48253

theorem right_triangle_median_length (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 :=
by sorry

end right_triangle_median_length_l482_48253


namespace quadratic_roots_sixth_power_sum_l482_48237

theorem quadratic_roots_sixth_power_sum (r s : ℝ) : 
  r^2 - 3 * r * Real.sqrt 2 + 4 = 0 ∧ 
  s^2 - 3 * s * Real.sqrt 2 + 4 = 0 → 
  r^6 + s^6 = 648 := by
sorry

end quadratic_roots_sixth_power_sum_l482_48237


namespace train_journey_duration_l482_48278

-- Define the time type
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

-- Define the function to calculate time difference
def timeDifference (t1 t2 : Time) : Time :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  let diffMinutes := totalMinutes2 - totalMinutes1
  { hours := diffMinutes / 60, minutes := diffMinutes % 60 }

-- Theorem statement
theorem train_journey_duration :
  let departureTime := { hours := 9, minutes := 20 : Time }
  let arrivalTime := { hours := 11, minutes := 30 : Time }
  timeDifference departureTime arrivalTime = { hours := 2, minutes := 10 : Time } := by
  sorry


end train_journey_duration_l482_48278


namespace equation_solution_l482_48265

theorem equation_solution : ∃! x : ℝ, 13 + Real.sqrt (-4 + 5 * x * 3) = 14 := by sorry

end equation_solution_l482_48265


namespace output_after_year_formula_l482_48229

/-- Calculates the output after 12 months given an initial output and monthly growth rate -/
def outputAfterYear (a : ℝ) (p : ℝ) : ℝ := a * (1 + p) ^ 12

/-- Theorem stating that the output after 12 months is equal to a(1+p)^12 -/
theorem output_after_year_formula (a : ℝ) (p : ℝ) :
  outputAfterYear a p = a * (1 + p) ^ 12 := by sorry

end output_after_year_formula_l482_48229


namespace greatest_three_digit_divisible_by_3_6_4_l482_48268

theorem greatest_three_digit_divisible_by_3_6_4 : ∃ n : ℕ, 
  n = 984 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 3 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 6 = 0 ∧ m % 4 = 0 → m ≤ n :=
by sorry

end greatest_three_digit_divisible_by_3_6_4_l482_48268


namespace cube_root_of_27_l482_48204

theorem cube_root_of_27 : 
  {z : ℂ | z^3 = 27} = {3, (-3 + 3*Complex.I*Real.sqrt 3)/2, (-3 - 3*Complex.I*Real.sqrt 3)/2} := by
  sorry

end cube_root_of_27_l482_48204


namespace no_common_integers_satisfying_condition_l482_48203

theorem no_common_integers_satisfying_condition : 
  ¬∃ i : ℤ, 10 ≤ i ∧ i ≤ 30 ∧ i^2 - 5*i - 6 = 0 := by
  sorry

end no_common_integers_satisfying_condition_l482_48203


namespace tennis_players_count_l482_48218

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (both : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 18 →
  both = 9 →
  neither = 2 →
  ∃ tennis : ℕ, tennis = 19 ∧ 
    total = badminton + tennis - both + neither :=
by sorry

end tennis_players_count_l482_48218


namespace infinite_geometric_series_first_term_l482_48251

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := by
  sorry

end infinite_geometric_series_first_term_l482_48251


namespace sugar_consumption_calculation_l482_48236

/-- Given a price increase and consumption change, calculate the initial consumption -/
theorem sugar_consumption_calculation 
  (price_increase : ℝ) 
  (expenditure_increase : ℝ) 
  (new_consumption : ℝ) 
  (h1 : price_increase = 0.32)
  (h2 : expenditure_increase = 0.10)
  (h3 : new_consumption = 25) :
  ∃ (initial_consumption : ℝ), 
    initial_consumption = 75 ∧ 
    (1 + price_increase) * new_consumption = (1 + expenditure_increase) * initial_consumption :=
by sorry

end sugar_consumption_calculation_l482_48236


namespace custom_op_example_l482_48234

-- Define the custom operation
def custom_op (A B : ℕ) : ℕ := (A + 3) * (B - 2)

-- State the theorem
theorem custom_op_example : custom_op 12 17 = 225 := by
  sorry

end custom_op_example_l482_48234


namespace intersection_complement_equals_set_l482_48213

-- Define the universal set U
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 7}

-- Define set A
def A : Set ℤ := {1, 3, 5, 7}

-- Define set B
def B : Set ℤ := {2, 4, 5}

-- Theorem statement
theorem intersection_complement_equals_set : B ∩ (U \ A) = {2, 4} := by
  sorry

end intersection_complement_equals_set_l482_48213


namespace function_value_order_l482_48285

noncomputable def f (x : ℝ) := Real.log (abs (x - 2)) + x^2 - 4*x

theorem function_value_order :
  let a := f (Real.log 9 / Real.log 2)
  let b := f (Real.log 18 / Real.log 4)
  let c := f 1
  a > c ∧ c > b :=
by sorry

end function_value_order_l482_48285


namespace min_difference_of_bounds_l482_48201

-- Define the arithmetic-geometric sequence
def a (n : ℕ) : ℚ := (4/3) * (-1/3)^(n-1)

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := 1 - (-1/3)^n

-- Define the function f(n) = S(n) - 1/S(n)
def f (n : ℕ) : ℚ := S n - 1 / (S n)

-- Theorem statement
theorem min_difference_of_bounds (A B : ℚ) :
  (∀ n : ℕ, n ≥ 1 → A ≤ f n ∧ f n ≤ B) →
  B - A ≥ 59/72 :=
sorry

end min_difference_of_bounds_l482_48201


namespace simplify_sqrt_expression_l482_48250

theorem simplify_sqrt_expression (x : ℝ) (h : x < 1) :
  (x - 1) * Real.sqrt (-1 / (x - 1)) = -Real.sqrt (1 - x) := by
  sorry

end simplify_sqrt_expression_l482_48250


namespace remainder_17_49_mod_5_l482_48241

theorem remainder_17_49_mod_5 : 17^49 % 5 = 2 := by
  sorry

end remainder_17_49_mod_5_l482_48241


namespace pairwise_product_inequality_l482_48292

theorem pairwise_product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * Real.rpow (x^3 + y^3 + z^3) (1/3) := by
  sorry

end pairwise_product_inequality_l482_48292


namespace solve_for_y_l482_48226

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - 3 * y = 9) (h2 : x + y = 8) : y = 1.4 := by
  sorry

end solve_for_y_l482_48226


namespace store_customer_ratio_l482_48293

theorem store_customer_ratio : 
  let non_holiday_rate : ℚ := 175  -- customers per hour during non-holiday season
  let holiday_total : ℕ := 2800    -- total customers during holiday season
  let holiday_hours : ℕ := 8       -- number of hours during holiday season
  let holiday_rate : ℚ := holiday_total / holiday_hours  -- customers per hour during holiday season
  holiday_rate / non_holiday_rate = 2 := by
sorry

end store_customer_ratio_l482_48293


namespace sandy_worked_five_days_l482_48288

/-- The number of days Sandy worked -/
def days_worked (total_hours : ℕ) (hours_per_day : ℕ) : ℚ :=
  total_hours / hours_per_day

/-- Proof that Sandy worked 5 days -/
theorem sandy_worked_five_days (total_hours : ℕ) (hours_per_day : ℕ) 
  (h1 : total_hours = 45)
  (h2 : hours_per_day = 9) :
  days_worked total_hours hours_per_day = 5 := by
  sorry

end sandy_worked_five_days_l482_48288


namespace original_group_size_l482_48280

theorem original_group_size (initial_avg : ℝ) (new_boy1 new_boy2 new_boy3 : ℝ) (new_avg : ℝ) :
  initial_avg = 35 →
  new_boy1 = 40 →
  new_boy2 = 45 →
  new_boy3 = 50 →
  new_avg = 36 →
  ∃ n : ℕ,
    n * initial_avg + new_boy1 + new_boy2 + new_boy3 = (n + 3) * new_avg ∧
    n = 27 :=
by sorry

end original_group_size_l482_48280


namespace store_sales_l482_48297

theorem store_sales (dvd_count : ℕ) (dvd_cd_ratio : ℚ) : 
  dvd_count = 168 → dvd_cd_ratio = 1.6 → dvd_count + (dvd_count / dvd_cd_ratio).floor = 273 := by
  sorry

end store_sales_l482_48297


namespace intersection_right_angle_coordinates_l482_48273

-- Define the line and parabola
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B as intersections
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Define point C on the parabola
def point_on_parabola (C : ℝ × ℝ) : Prop :=
  parabola C.1 C.2

-- Define the right angle condition
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem intersection_right_angle_coordinates :
  ∀ A B C : ℝ × ℝ,
  intersection_points A B →
  point_on_parabola C →
  right_angle A B C →
  (C = (1, -2) ∨ C = (9, -6)) :=
sorry

end intersection_right_angle_coordinates_l482_48273


namespace quadratic_monotonicity_implies_a_range_l482_48217

/-- A function f is monotonic on an interval [a, b] if it is either
    nondecreasing or nonincreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

theorem quadratic_monotonicity_implies_a_range (a : ℝ) :
  IsMonotonic (fun x => x^2 - 2*a*x - 3) 1 2 → a ≤ 1 ∨ a ≥ 2 := by
  sorry

end quadratic_monotonicity_implies_a_range_l482_48217


namespace divisible_by_13_with_sqrt_between_24_and_24_5_verify_585_and_598_l482_48202

theorem divisible_by_13_with_sqrt_between_24_and_24_5 : 
  ∃ (n : ℕ), n > 0 ∧ n % 13 = 0 ∧ 24 < Real.sqrt n ∧ Real.sqrt n < 24.5 :=
by
  sorry

theorem verify_585_and_598 : 
  (585 > 0 ∧ 585 % 13 = 0 ∧ 24 < Real.sqrt 585 ∧ Real.sqrt 585 < 24.5) ∧
  (598 > 0 ∧ 598 % 13 = 0 ∧ 24 < Real.sqrt 598 ∧ Real.sqrt 598 < 24.5) :=
by
  sorry

end divisible_by_13_with_sqrt_between_24_and_24_5_verify_585_and_598_l482_48202


namespace find_number_l482_48267

theorem find_number : ∃! x : ℝ, 0.8 * x + 20 = x := by
  sorry

end find_number_l482_48267


namespace cos_function_identity_l482_48221

theorem cos_function_identity (f : ℝ → ℝ) (x : ℝ) 
  (h : ∀ x, f (Real.sin x) = 2 - Real.cos (2 * x)) : 
  f (Real.cos x) = 2 + Real.cos x ^ 2 := by
  sorry

end cos_function_identity_l482_48221


namespace circle_triangle_area_difference_l482_48206

/-- Given an equilateral triangle with side length 12 units and its circumscribed circle,
    the difference between the area of the circle and the area of the triangle
    is 144π - 36√3 square units. -/
theorem circle_triangle_area_difference : 
  let s : ℝ := 12 -- side length of the equilateral triangle
  let r : ℝ := s -- radius of the circumscribed circle (equal to side length)
  let circle_area : ℝ := π * r^2
  let triangle_height : ℝ := s * (Real.sqrt 3) / 2
  let triangle_area : ℝ := (1/2) * s * triangle_height
  circle_area - triangle_area = 144 * π - 36 * Real.sqrt 3 :=
by sorry

end circle_triangle_area_difference_l482_48206


namespace units_digit_of_2015_powers_l482_48227

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that a number ends with 5 -/
def ends_with_5 (n : ℕ) : Prop := units_digit n = 5

/-- The property that powers of numbers ending in 5 always end in 5 for exponents ≥ 1 -/
def power_ends_with_5 (n : ℕ) : Prop := 
  ends_with_5 n → ∀ k : ℕ, k ≥ 1 → ends_with_5 (n^k)

theorem units_digit_of_2015_powers : 
  ends_with_5 2015 → 
  power_ends_with_5 2015 → 
  units_digit (2015^2 + 2015^0 + 2015^1 + 2015^5) = 6 := by sorry

end units_digit_of_2015_powers_l482_48227


namespace odd_function_property_l482_48284

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x - 3) = f (x + 2))
  (h_value : f 1 = 2) :
  f 2011 - f 2010 = 2 := by
sorry

end odd_function_property_l482_48284


namespace fraction_equality_l482_48289

theorem fraction_equality : (4 + 14) / (7 + 14) = 6 / 7 := by sorry

end fraction_equality_l482_48289


namespace greatest_power_of_three_in_19_factorial_l482_48296

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Counts the number of factors of 3 in n! -/
def count_factors_of_three (n : ℕ) : ℕ :=
  if n < 3 then 0
  else (n / 3) + count_factors_of_three (n / 3)

theorem greatest_power_of_three_in_19_factorial :
  ∀ n : ℕ, 3^n ∣ factorial 19 ↔ n ≤ 8 :=
sorry

end greatest_power_of_three_in_19_factorial_l482_48296


namespace inequality_solution_set_l482_48283

theorem inequality_solution_set (x : ℝ) :
  (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by
  sorry

end inequality_solution_set_l482_48283


namespace tangent_slope_at_point_2_10_l482_48244

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_point_2_10 :
  f_derivative 2 = 7 :=
sorry

end tangent_slope_at_point_2_10_l482_48244


namespace problem_1_problem_2_problem_3_l482_48242

-- Problem 1
theorem problem_1 : (1) - 2^2 + (π - 3)^0 + 0.5^(-1) = -1 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (x - 2*y) * (x^2 + 2*x*y + 4*y^2) = x^3 - 8*y^3 := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) : a * a^2 * a^3 + (-2*a^3)^2 - a^8 / a^2 = 4*a^6 := by sorry

end problem_1_problem_2_problem_3_l482_48242


namespace function_properties_l482_48259

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_neg : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ (f 2 = f 0) := by sorry

end function_properties_l482_48259


namespace unique_initial_pair_l482_48274

def arithmetic_mean_operation (a b : ℕ) : ℕ × ℕ :=
  if (a + b) % 2 = 0 then
    let mean := (a + b) / 2
    if mean < a then (mean, a) else (b, mean)
  else
    (a, b)

def perform_operations (n : ℕ) (pair : ℕ × ℕ) : ℕ × ℕ :=
  match n with
  | 0 => pair
  | n + 1 => perform_operations n (arithmetic_mean_operation pair.1 pair.2)

theorem unique_initial_pair :
  ∀ x : ℕ,
    x < 2015 →
    x ≠ 991 →
    ∃ i : ℕ,
      i ≤ 10 ∧
      (perform_operations i (x, 2015)).1 = (perform_operations i (x, 2015)).2 :=
sorry

end unique_initial_pair_l482_48274


namespace circumscribed_sphere_radius_hexagonal_pyramid_l482_48224

/-- The radius of a sphere circumscribed around a regular hexagonal pyramid -/
theorem circumscribed_sphere_radius_hexagonal_pyramid 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  (h₃ : a < b) : 
  ∃ R : ℝ, R = b^2 / (2 * Real.sqrt (b^2 - a^2)) ∧ 
  R > 0 ∧
  R * 2 * Real.sqrt (b^2 - a^2) = b^2 :=
sorry

end circumscribed_sphere_radius_hexagonal_pyramid_l482_48224


namespace polynomial_factor_coefficients_l482_48240

theorem polynomial_factor_coefficients : 
  ∃ (a b : ℤ), 
    (∃ (d : ℤ), 3 * X ^ 4 + b * X ^ 3 + 45 * X ^ 2 - 21 * X + 8 = 
      (2 * X ^ 2 - 3 * X + 2) * (a * X ^ 2 + d * X + 4)) ∧ 
    a = 3 ∧ 
    b = -27 := by
  sorry

end polynomial_factor_coefficients_l482_48240


namespace frame_purchase_remaining_money_l482_48249

theorem frame_purchase_remaining_money 
  (budget : ℝ) 
  (initial_frame_price_increase : ℝ) 
  (smaller_frame_price_ratio : ℝ) :
  budget = 60 →
  initial_frame_price_increase = 0.2 →
  smaller_frame_price_ratio = 3/4 →
  budget - (budget * (1 + initial_frame_price_increase) * smaller_frame_price_ratio) = 6 := by
  sorry

end frame_purchase_remaining_money_l482_48249


namespace tetrahedron_projection_areas_l482_48269

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the xOy plane -/
def projectionAreaXOY (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the yOz plane -/
def projectionAreaYOZ (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the zOx plane -/
def projectionAreaZOX (p1 p2 p3 : Point3D) : ℝ := sorry

theorem tetrahedron_projection_areas :
  let A : Point3D := ⟨2, 0, 0⟩
  let B : Point3D := ⟨2, 2, 0⟩
  let C : Point3D := ⟨0, 2, 0⟩
  let D : Point3D := ⟨1, 1, Real.sqrt 2⟩
  let S₁ := projectionAreaXOY A B C + projectionAreaXOY A B D + projectionAreaXOY A C D + projectionAreaXOY B C D
  let S₂ := projectionAreaYOZ A B C + projectionAreaYOZ A B D + projectionAreaYOZ A C D + projectionAreaYOZ B C D
  let S₃ := projectionAreaZOX A B C + projectionAreaZOX A B D + projectionAreaZOX A C D + projectionAreaZOX B C D
  S₃ = S₂ ∧ S₃ ≠ S₁ := by sorry

end tetrahedron_projection_areas_l482_48269


namespace total_heads_count_l482_48266

/-- The number of feet per hen -/
def henFeet : ℕ := 2

/-- The number of feet per cow -/
def cowFeet : ℕ := 4

/-- Theorem: Given a group of hens and cows, if the total number of feet is 140
    and there are 26 hens, then the total number of heads is 48. -/
theorem total_heads_count (totalFeet : ℕ) (henCount : ℕ) : 
  totalFeet = 140 → henCount = 26 → henCount * henFeet + (totalFeet - henCount * henFeet) / cowFeet = 48 := by
  sorry

end total_heads_count_l482_48266


namespace largest_sum_of_3digit_numbers_l482_48231

def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (∃ (d1 d2 d3 d4 d5 d6 : Nat),
    {d1, d2, d3, d4, d5, d6} = digits ∧
    a = 100 * d1 + 10 * d2 + d3 ∧
    b = 100 * d4 + 10 * d5 + d6)

def sum_of_pair (a b : Nat) : Nat := a + b

theorem largest_sum_of_3digit_numbers :
  (∃ (a b : Nat), is_valid_pair a b ∧
    ∀ (x y : Nat), is_valid_pair x y → sum_of_pair x y ≤ sum_of_pair a b) ∧
  (∀ (a b : Nat), is_valid_pair a b → sum_of_pair a b ≤ 1803) ∧
  (∃ (a b : Nat), is_valid_pair a b ∧ sum_of_pair a b = 1803) := by
  sorry

end largest_sum_of_3digit_numbers_l482_48231


namespace valid_arrangements_ten_four_l482_48261

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k people in a row within a group -/
def groupArrangements (k : ℕ) : ℕ := Nat.factorial k

/-- The number of ways to arrange n people in a row, where k specific people are not allowed to sit in k consecutive seats -/
def validArrangements (n k : ℕ) : ℕ :=
  totalArrangements n - totalArrangements (n - k + 1) * groupArrangements k

theorem valid_arrangements_ten_four :
  validArrangements 10 4 = 3507840 := by sorry

end valid_arrangements_ten_four_l482_48261


namespace arithmetic_calculations_l482_48233

-- Define the arithmetic operations
def calculation1 : ℤ := 36 * 17 + 129
def calculation2 : ℤ := 320 * (300 - 294)
def calculation3 : ℤ := 25 * 5 * 4
def calculation4 : ℚ := 18.45 - 25.6 - 24.4

-- Theorem statements
theorem arithmetic_calculations :
  (calculation1 = 741) ∧
  (calculation2 = 1920) ∧
  (calculation3 = 500) ∧
  (calculation4 = -31.55) := by
  sorry

end arithmetic_calculations_l482_48233


namespace pentagon_reconstruction_l482_48235

noncomputable section

-- Define the vector space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points of the pentagon and extended points
variable (A B C D E A' B' C' D' E' : V)

-- Define the conditions
variable (h1 : A' - A = A - B)
variable (h2 : B' - B = B - C)
variable (h3 : C' - C = C - D)
variable (h4 : D' - D = D - E)
variable (h5 : E' - E = E - A)

-- State the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (2/31 : ℝ) • B' + (4/31 : ℝ) • C' + (8/31 : ℝ) • D' + (16/31 : ℝ) • E' :=
sorry

end pentagon_reconstruction_l482_48235


namespace one_third_vector_AB_l482_48287

/-- Given two vectors OA and OB in 2D space, prove that 1/3 of vector AB equals the specified result. -/
theorem one_third_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (4, 8) → OB = (-7, -2) → (1 / 3 : ℝ) • (OB - OA) = (-11/3, -10/3) := by
  sorry

end one_third_vector_AB_l482_48287


namespace orange_count_correct_l482_48277

/-- The number of oranges in the box after adding and removing specified quantities -/
def final_oranges (initial added removed : ℝ) : ℝ :=
  initial + added - removed

/-- Theorem stating that the final number of oranges in the box is correct -/
theorem orange_count_correct (initial added removed : ℝ) :
  final_oranges initial added removed = initial + added - removed := by
  sorry

end orange_count_correct_l482_48277


namespace solution_set_f_range_of_m_l482_48270

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x| + |x + 1|

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f : 
  {x : ℝ | f x > 3} = {x : ℝ | x > 1 ∨ x < -2} := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  (∀ x : ℝ, m^2 + 3*m + 2*f x ≥ 0) → (m ≥ -1 ∨ m ≤ -2) := by sorry

end solution_set_f_range_of_m_l482_48270


namespace fox_catches_rabbits_l482_48225

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the game setup -/
structure GameSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  foxSpeed : ℝ
  rabbitSpeed : ℝ

/-- Checks if the fox can catch both rabbits -/
def canCatchBothRabbits (setup : GameSetup) : Prop :=
  setup.foxSpeed ≥ 1 + Real.sqrt 2

theorem fox_catches_rabbits (setup : GameSetup) 
  (h1 : setup.A = ⟨0, 0⟩) 
  (h2 : setup.B = ⟨1, 0⟩) 
  (h3 : setup.C = ⟨1, 1⟩) 
  (h4 : setup.D = ⟨0, 1⟩)
  (h5 : setup.rabbitSpeed = 1) :
  canCatchBothRabbits setup ↔ 
    ∀ (t : ℝ), t ≥ 0 → 
      ∃ (foxPos : Point),
        (foxPos.x - setup.C.x)^2 + (foxPos.y - setup.C.y)^2 ≤ (setup.foxSpeed * t)^2 ∧
        ((foxPos.x = setup.B.x + t ∧ foxPos.y = 0) ∨
         (foxPos.x = 0 ∧ foxPos.y = setup.D.y + t) ∨
         (foxPos.x = 0 ∧ foxPos.y = 0)) :=
by sorry

end fox_catches_rabbits_l482_48225


namespace composition_of_even_is_even_l482_48222

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end composition_of_even_is_even_l482_48222


namespace no_valid_grid_l482_48299

/-- Represents a 4x4 grid with some initial values -/
structure Grid :=
  (a11 : ℝ) (a12 : ℝ) (a13 : ℝ) (a14 : ℝ)
  (a21 : ℝ) (a22 : ℝ) (a23 : ℝ) (a24 : ℝ)
  (a31 : ℝ) (a32 : ℝ) (a33 : ℝ) (a34 : ℝ)
  (a41 : ℝ) (a42 : ℝ) (a43 : ℝ) (a44 : ℝ)

/-- Checks if a sequence of 4 numbers forms an arithmetic progression -/
def isArithmeticSequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

/-- Defines the conditions for the grid based on the problem statement -/
def validGrid (g : Grid) : Prop :=
  g.a12 = 9 ∧ g.a21 = 1 ∧ g.a34 = 5 ∧ g.a43 = 8 ∧
  isArithmeticSequence g.a11 g.a12 g.a13 g.a14 ∧
  isArithmeticSequence g.a21 g.a22 g.a23 g.a24 ∧
  isArithmeticSequence g.a31 g.a32 g.a33 g.a34 ∧
  isArithmeticSequence g.a41 g.a42 g.a43 g.a44 ∧
  isArithmeticSequence g.a11 g.a21 g.a31 g.a41 ∧
  isArithmeticSequence g.a12 g.a22 g.a32 g.a42 ∧
  isArithmeticSequence g.a13 g.a23 g.a33 g.a43 ∧
  isArithmeticSequence g.a14 g.a24 g.a34 g.a44

/-- The main theorem stating that no valid grid exists -/
theorem no_valid_grid : ¬ ∃ (g : Grid), validGrid g := by
  sorry

end no_valid_grid_l482_48299


namespace trip_time_difference_l482_48295

theorem trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) :
  speed = 60 → distance1 = 540 → distance2 = 570 →
  (distance2 - distance1) / speed * 60 = 30 := by
  sorry

end trip_time_difference_l482_48295


namespace sandy_book_purchase_l482_48276

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop -/
def amount_first_shop : ℚ := 1380

/-- The amount Sandy spent at the second shop -/
def amount_second_shop : ℚ := 900

/-- The average price Sandy paid per book -/
def average_price : ℚ := 19

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

theorem sandy_book_purchase :
  (amount_first_shop + amount_second_shop) / (books_first_shop + books_second_shop : ℚ) = average_price :=
by sorry

end sandy_book_purchase_l482_48276


namespace quarter_circle_arcs_sum_l482_48272

/-- The sum of the lengths of n quarter-circle arcs, each constructed on a segment of length D/n 
    (where D is the diameter of a large circle), approaches πD/8 as n approaches infinity. -/
theorem quarter_circle_arcs_sum (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * (D / n) / 4) - π * D / 8| < ε :=
sorry

end quarter_circle_arcs_sum_l482_48272


namespace number_problem_l482_48245

theorem number_problem (x : ℝ) : (0.16 * (0.40 * x) = 6) → x = 93.75 := by
  sorry

end number_problem_l482_48245


namespace product_positive_l482_48258

theorem product_positive (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b * (a - b) > 0 := by
  sorry

end product_positive_l482_48258


namespace actual_daily_length_is_72_required_daily_increase_at_least_36_l482_48223

/-- Represents the renovation of a pipe network --/
structure PipeRenovation where
  totalLength : ℝ
  originalDailyLength : ℝ
  efficiencyIncrease : ℝ
  daysAheadOfSchedule : ℝ
  constructedDays : ℝ
  maxTotalDays : ℝ

/-- Calculates the actual daily renovation length --/
def actualDailyLength (pr : PipeRenovation) : ℝ :=
  pr.originalDailyLength * (1 + pr.efficiencyIncrease)

/-- Theorem for the actual daily renovation length --/
theorem actual_daily_length_is_72 (pr : PipeRenovation)
  (h1 : pr.totalLength = 3600)
  (h2 : pr.efficiencyIncrease = 0.2)
  (h3 : pr.daysAheadOfSchedule = 10)
  (h4 : pr.totalLength / pr.originalDailyLength - pr.totalLength / (actualDailyLength pr) = pr.daysAheadOfSchedule) :
  actualDailyLength pr = 72 := by sorry

/-- Theorem for the required increase in daily renovation length --/
theorem required_daily_increase_at_least_36 (pr : PipeRenovation)
  (h1 : pr.totalLength = 3600)
  (h2 : actualDailyLength pr = 72)
  (h3 : pr.constructedDays = 20)
  (h4 : pr.maxTotalDays = 40) :
  ∃ m : ℝ, m ≥ 36 ∧ (pr.maxTotalDays - pr.constructedDays) * (actualDailyLength pr + m) ≥ pr.totalLength - actualDailyLength pr * pr.constructedDays := by sorry

end actual_daily_length_is_72_required_daily_increase_at_least_36_l482_48223


namespace centroid_count_l482_48281

/-- A point on the perimeter of the square -/
structure PerimeterPoint where
  x : ℚ
  y : ℚ
  on_perimeter : (x = 0 ∨ x = 12) ∨ (y = 0 ∨ y = 12)
  valid_coord : (0 ≤ x ∧ x ≤ 12) ∧ (0 ≤ y ∧ y ≤ 12)

/-- The set of 48 equally spaced points on the perimeter -/
def perimeter_points : Finset PerimeterPoint :=
  sorry

/-- Predicate to check if two points are consecutive on the perimeter -/
def are_consecutive (p q : PerimeterPoint) : Prop :=
  sorry

/-- The centroid of a triangle given by three points -/
def centroid (p q r : PerimeterPoint) : ℚ × ℚ :=
  ((p.x + q.x + r.x) / 3, (p.y + q.y + r.y) / 3)

/-- The set of all possible centroids -/
def possible_centroids : Finset (ℚ × ℚ) :=
  sorry

theorem centroid_count :
  ∀ p q r : PerimeterPoint,
    p ∈ perimeter_points →
    q ∈ perimeter_points →
    r ∈ perimeter_points →
    ¬(are_consecutive p q ∨ are_consecutive q r ∨ are_consecutive r p) →
    (Finset.card possible_centroids = 1156) :=
  sorry

end centroid_count_l482_48281


namespace infinite_solutions_when_m_is_two_l482_48200

theorem infinite_solutions_when_m_is_two :
  ∃ (m : ℝ), ∀ (x : ℝ), m^2 * x + m * (1 - x) - 2 * (1 + x) = 0 → 
  (m = 2 ∧ ∀ (y : ℝ), m^2 * y + m * (1 - y) - 2 * (1 + y) = 0) :=
by sorry

end infinite_solutions_when_m_is_two_l482_48200


namespace functional_equation_solution_l482_48238

/-- The functional equation that f must satisfy -/
def functional_equation (f : ℝ → ℝ) (α β : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x * f y = y^α * f (x/2) + x^β * f (y/2)

/-- The theorem stating the possible forms of f -/
theorem functional_equation_solution (f : ℝ → ℝ) (α β : ℝ) :
  functional_equation f α β →
  (∃ c : ℝ, c = 2^(1-α) ∧ ∀ x, x > 0 → f x = c * x^α) ∨
  (∀ x, x > 0 → f x = 0) := by
  sorry

end functional_equation_solution_l482_48238


namespace investment_proof_l482_48279

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_proof :
  let initial_investment : ℝ := 1000
  let interest_rate : ℝ := 0.08
  let time_period : ℕ := 6
  let final_balance : ℝ := 1586.87
  (compound_interest initial_investment interest_rate time_period) = final_balance := by
  sorry

end investment_proof_l482_48279


namespace coefficient_x5y_in_expansion_l482_48211

/-- The coefficient of x^5y in the expansion of (x-2y)^5(x+y) -/
def coefficient_x5y : ℤ := -9

/-- The expansion of (x-2y)^5(x+y) -/
def expansion (x y : ℚ) : ℚ := (x - 2*y)^5 * (x + y)

theorem coefficient_x5y_in_expansion :
  coefficient_x5y = (
    -- Extract the coefficient of x^5y from the expansion
    -- This part is left unimplemented as it requires complex polynomial manipulation
    sorry
  ) := by sorry

end coefficient_x5y_in_expansion_l482_48211


namespace range_of_a_l482_48210

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → a < x + 1/x) → a < 2 := by sorry

end range_of_a_l482_48210


namespace largest_divisible_power_of_three_l482_48282

theorem largest_divisible_power_of_three : ∃! n : ℕ, 
  (∀ k : ℕ, k ≤ n → (4^27000 - 82) % 3^k = 0) ∧
  (∀ m : ℕ, m > n → (4^27000 - 82) % 3^m ≠ 0) ∧
  n = 5 :=
by sorry

end largest_divisible_power_of_three_l482_48282


namespace average_of_last_three_l482_48264

theorem average_of_last_three (A B C D : ℝ) : 
  A = 33 →
  D = 18 →
  (A + B + C) / 3 = 20 →
  (B + C + D) / 3 = 15 := by
sorry

end average_of_last_three_l482_48264


namespace B_max_at_181_l482_48255

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 2000 k) * (0.1 ^ k)

/-- The theorem stating that B_k is maximum when k = 181 -/
theorem B_max_at_181 : ∀ k ∈ Finset.range 2001, B 181 ≥ B k := by sorry

end B_max_at_181_l482_48255
