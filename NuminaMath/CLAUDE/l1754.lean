import Mathlib

namespace NUMINAMATH_CALUDE_number_subtraction_problem_l1754_175455

theorem number_subtraction_problem : ∃! x : ℝ, 0.4 * x - 11 = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l1754_175455


namespace NUMINAMATH_CALUDE_odd_function_extension_l1754_175420

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension
  (f : ℝ → ℝ)
  (odd : is_odd f)
  (pos_def : ∀ x > 0, f x = x - 1) :
  ∀ x < 0, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1754_175420


namespace NUMINAMATH_CALUDE_haleys_marbles_l1754_175459

/-- The number of boys in Haley's class who love to play marbles. -/
def num_boys : ℕ := 5

/-- The number of marbles each boy would receive. -/
def marbles_per_boy : ℕ := 7

/-- The total number of marbles Haley has. -/
def total_marbles : ℕ := num_boys * marbles_per_boy

/-- Theorem stating that the total number of marbles Haley has is equal to
    the product of the number of boys and the number of marbles each boy would receive. -/
theorem haleys_marbles : total_marbles = num_boys * marbles_per_boy := by
  sorry

end NUMINAMATH_CALUDE_haleys_marbles_l1754_175459


namespace NUMINAMATH_CALUDE_one_third_percent_of_150_l1754_175449

theorem one_third_percent_of_150 : (1 / 3 : ℚ) / 100 * 150 = 0.5 := by sorry

end NUMINAMATH_CALUDE_one_third_percent_of_150_l1754_175449


namespace NUMINAMATH_CALUDE_lucia_outfits_l1754_175456

/-- Represents the number of different outfits Lucia can create -/
def outfits (shoes dresses hats : ℕ) : ℕ := shoes * dresses * hats

/-- Proves that Lucia can create 60 different outfits -/
theorem lucia_outfits :
  outfits 3 5 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lucia_outfits_l1754_175456


namespace NUMINAMATH_CALUDE_M_characterization_a_range_l1754_175419

-- Define the set M
def M : Set ℝ := {m | ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

-- Define the set N
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x + a - 2) < 0}

-- Statement 1
theorem M_characterization : M = {m | -1/4 ≤ m ∧ m < 2} := by sorry

-- Statement 2
theorem a_range (h : ∀ m ∈ M, ∃ x ∈ N a, x^2 - x - m = 0) : 
  a ∈ Set.Iic (-1/4) ∪ Set.Ioi (9/4) := by sorry

end NUMINAMATH_CALUDE_M_characterization_a_range_l1754_175419


namespace NUMINAMATH_CALUDE_least_possible_b_l1754_175488

-- Define Fibonacci sequence
def isFibonacci (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (((1 + Real.sqrt 5) / 2) ^ k - ((1 - Real.sqrt 5) / 2) ^ k) / Real.sqrt 5

-- Define the problem
theorem least_possible_b (a b : ℕ) : 
  (a + b = 90) →  -- Sum of acute angles in a right triangle
  (a > b) →       -- a is greater than b
  isFibonacci a → -- a is a Fibonacci number
  isFibonacci b → -- b is a Fibonacci number
  (∀ c : ℕ, c < b → (c + a ≠ 90 ∨ ¬isFibonacci c ∨ ¬isFibonacci a)) →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_b_l1754_175488


namespace NUMINAMATH_CALUDE_set_conditions_equivalence_l1754_175434

theorem set_conditions_equivalence (m : ℝ) :
  let A := {x : ℝ | 0 < x - m ∧ x - m < 2}
  let B := {x : ℝ | -x^2 + 3*x ≤ 0}
  (A ∩ B = ∅ ∧ A ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_set_conditions_equivalence_l1754_175434


namespace NUMINAMATH_CALUDE_files_deleted_l1754_175428

theorem files_deleted (initial_files initial_apps final_files final_apps : ℕ) 
  (h1 : initial_files = 24)
  (h2 : initial_apps = 13)
  (h3 : final_files = 21)
  (h4 : final_apps = 17) :
  initial_files - final_files = 3 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l1754_175428


namespace NUMINAMATH_CALUDE_candy_problem_l1754_175468

theorem candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies / 2
  let day2_remaining := day1_remaining / 3 * 2
  let day3_remaining := day2_remaining / 4 * 3
  let day4_remaining := day3_remaining / 5 * 4
  let day5_remaining := day4_remaining / 6 * 5
  day5_remaining = 1 → initial_candies = 720 :=
by sorry

end NUMINAMATH_CALUDE_candy_problem_l1754_175468


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1754_175458

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  b = 2 * a →        -- one leg is twice the other
  a^2 + b^2 + c^2 = 1450 →  -- sum of squares of sides
  c = 5 * Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1754_175458


namespace NUMINAMATH_CALUDE_oblomov_weight_change_l1754_175473

theorem oblomov_weight_change : 
  let spring_factor : ℝ := 0.75
  let summer_factor : ℝ := 1.20
  let autumn_factor : ℝ := 0.90
  let winter_factor : ℝ := 1.20
  spring_factor * summer_factor * autumn_factor * winter_factor < 1 := by
sorry

end NUMINAMATH_CALUDE_oblomov_weight_change_l1754_175473


namespace NUMINAMATH_CALUDE_sheep_wool_production_l1754_175438

/-- Calculates the amount of wool produced per sheep given the total number of sheep,
    payment to the shearer, price per pound of wool, and total profit. -/
def wool_per_sheep (num_sheep : ℕ) (shearer_payment : ℕ) (price_per_pound : ℕ) (profit : ℕ) : ℕ :=
  ((profit + shearer_payment) / price_per_pound) / num_sheep

/-- Proves that given 200 sheep, $2000 paid to shearer, $20 per pound of wool,
    and $38000 profit, each sheep produces 10 pounds of wool. -/
theorem sheep_wool_production :
  wool_per_sheep 200 2000 20 38000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sheep_wool_production_l1754_175438


namespace NUMINAMATH_CALUDE_sin_sum_pi_minus_plus_alpha_l1754_175437

theorem sin_sum_pi_minus_plus_alpha (α : ℝ) : 
  Real.sin (π - α) + Real.sin (π + α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_pi_minus_plus_alpha_l1754_175437


namespace NUMINAMATH_CALUDE_lcm_18_24_l1754_175436

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1754_175436


namespace NUMINAMATH_CALUDE_total_pages_calculation_l1754_175490

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 49

/-- The total number of pages in all booklets -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem total_pages_calculation :
  total_pages = 441 :=
by sorry

end NUMINAMATH_CALUDE_total_pages_calculation_l1754_175490


namespace NUMINAMATH_CALUDE_least_candies_to_remove_daniel_candy_problem_l1754_175493

theorem least_candies_to_remove (total_candies : Nat) (sisters : Nat) : Nat :=
  let remainder := total_candies % sisters
  if remainder = 0 then 0 else sisters - remainder

theorem daniel_candy_problem :
  least_candies_to_remove 25 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_candies_to_remove_daniel_candy_problem_l1754_175493


namespace NUMINAMATH_CALUDE_leftover_value_l1754_175480

/-- The number of quarters in a roll -/
def quarters_per_roll : ℕ := 50

/-- The number of dimes in a roll -/
def dimes_per_roll : ℕ := 40

/-- The number of quarters Kim has -/
def kim_quarters : ℕ := 95

/-- The number of dimes Kim has -/
def kim_dimes : ℕ := 183

/-- The number of quarters Mark has -/
def mark_quarters : ℕ := 157

/-- The number of dimes Mark has -/
def mark_dimes : ℕ := 328

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1 / 4

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The total value of leftover coins after making complete rolls -/
theorem leftover_value : 
  let total_quarters := kim_quarters + mark_quarters
  let total_dimes := kim_dimes + mark_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_l1754_175480


namespace NUMINAMATH_CALUDE_reciprocal_sum_l1754_175446

theorem reciprocal_sum (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 56) :
  1 / x + 1 / y = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l1754_175446


namespace NUMINAMATH_CALUDE_initial_puppies_count_l1754_175499

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left -/
def puppies_left : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given_away + puppies_left

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l1754_175499


namespace NUMINAMATH_CALUDE_point_distance_and_reflection_l1754_175417

/-- Point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point2D) : ℝ := |p.x|

/-- Reflection of a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem point_distance_and_reflection :
  let P : Point2D := { x := 4, y := -2 }
  (distanceToYAxis P = 4) ∧
  (reflectAcrossXAxis P = { x := 4, y := 2 }) := by
  sorry

end NUMINAMATH_CALUDE_point_distance_and_reflection_l1754_175417


namespace NUMINAMATH_CALUDE_coles_return_speed_l1754_175496

/-- Calculates the average speed for the return trip given the conditions of Cole's journey -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : 
  speed_to_work = 60 → 
  total_time = 2 → 
  time_to_work = 1.2 → 
  (speed_to_work * time_to_work) / (total_time - time_to_work) = 90 := by
sorry

end NUMINAMATH_CALUDE_coles_return_speed_l1754_175496


namespace NUMINAMATH_CALUDE_sin_negative_2055_degrees_l1754_175475

theorem sin_negative_2055_degrees : 
  Real.sin ((-2055 : ℝ) * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_2055_degrees_l1754_175475


namespace NUMINAMATH_CALUDE_gold_bar_ratio_l1754_175412

theorem gold_bar_ratio (initial_bars : ℕ) (tax_percent : ℚ) (final_bars : ℕ) : 
  initial_bars = 60 →
  tax_percent = 1/10 →
  final_bars = 27 →
  (initial_bars - initial_bars * tax_percent - final_bars) / (initial_bars - initial_bars * tax_percent) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_gold_bar_ratio_l1754_175412


namespace NUMINAMATH_CALUDE_yellow_leaves_count_l1754_175448

theorem yellow_leaves_count (thursday_leaves friday_leaves saturday_leaves : ℕ)
  (thursday_brown_percent thursday_green_percent : ℚ)
  (friday_brown_percent friday_green_percent : ℚ)
  (saturday_brown_percent saturday_green_percent : ℚ)
  (h1 : thursday_leaves = 15)
  (h2 : friday_leaves = 22)
  (h3 : saturday_leaves = 30)
  (h4 : thursday_brown_percent = 25/100)
  (h5 : thursday_green_percent = 40/100)
  (h6 : friday_brown_percent = 30/100)
  (h7 : friday_green_percent = 20/100)
  (h8 : saturday_brown_percent = 15/100)
  (h9 : saturday_green_percent = 50/100) :
  ⌊thursday_leaves * (1 - thursday_brown_percent - thursday_green_percent)⌋ +
  ⌊friday_leaves * (1 - friday_brown_percent - friday_green_percent)⌋ +
  ⌊saturday_leaves * (1 - saturday_brown_percent - saturday_green_percent)⌋ = 26 := by
sorry

end NUMINAMATH_CALUDE_yellow_leaves_count_l1754_175448


namespace NUMINAMATH_CALUDE_smallest_valid_seating_eighteen_is_valid_smallest_seating_is_eighteen_l1754_175494

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  totalChairs : Nat
  seatedPeople : Nat

/-- Checks if a seating arrangement is valid (any new person must sit next to someone). -/
def isValidSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.totalChairs ≥ table.seatedPeople ∧
  table.totalChairs % table.seatedPeople = 0 ∧
  table.totalChairs / table.seatedPeople ≤ 4

/-- The theorem stating the smallest valid number of seated people for a 72-chair table. -/
theorem smallest_valid_seating :
  ∀ (table : CircularTable),
    table.totalChairs = 72 →
    isValidSeating table →
    table.seatedPeople ≥ 18 :=
by
  sorry

/-- The theorem stating that 18 is a valid seating arrangement for a 72-chair table. -/
theorem eighteen_is_valid :
  isValidSeating { totalChairs := 72, seatedPeople := 18 } :=
by
  sorry

/-- The main theorem combining the above results to prove 18 is the smallest valid seating. -/
theorem smallest_seating_is_eighteen :
  ∃ (table : CircularTable),
    table.totalChairs = 72 ∧
    table.seatedPeople = 18 ∧
    isValidSeating table ∧
    ∀ (otherTable : CircularTable),
      otherTable.totalChairs = 72 →
      isValidSeating otherTable →
      otherTable.seatedPeople ≥ table.seatedPeople :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_eighteen_is_valid_smallest_seating_is_eighteen_l1754_175494


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1754_175451

/-- Proves that a rectangle with perimeter 150 cm and length 15 cm greater than width has width 30 cm and length 45 cm -/
theorem rectangle_dimensions (w l : ℝ) 
  (h_perimeter : 2 * w + 2 * l = 150)
  (h_length_width : l = w + 15) :
  w = 30 ∧ l = 45 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1754_175451


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1754_175439

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 3 + a^(x - 1)
  f 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1754_175439


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1754_175498

theorem circle_intersection_range (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  1 ≤ m ∧ m ≤ 121 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1754_175498


namespace NUMINAMATH_CALUDE_quadrilateral_circumscription_l1754_175454

def can_be_circumscribed (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a + c = 180 ∧ b + d = 180

theorem quadrilateral_circumscription :
  (∃ (x : ℝ), can_be_circumscribed (2*x) (4*x) (5*x) (3*x)) ∧
  (∀ (x : ℝ), ¬can_be_circumscribed (5*x) (7*x) (8*x) (9*x)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_circumscription_l1754_175454


namespace NUMINAMATH_CALUDE_bacon_tomato_difference_l1754_175425

theorem bacon_tomato_difference (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 228)
  (h2 : bacon = 337)
  (h3 : tomatoes = 23) :
  bacon - tomatoes = 314 := by
  sorry

end NUMINAMATH_CALUDE_bacon_tomato_difference_l1754_175425


namespace NUMINAMATH_CALUDE_total_bathing_suits_l1754_175497

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l1754_175497


namespace NUMINAMATH_CALUDE_least_value_cubic_equation_l1754_175472

theorem least_value_cubic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^3 + 3 * y^2 + 5 * y + 1
  ∃ y_min : ℝ,
    f y_min = 5 ∧
    ∀ y : ℝ, f y = 5 → y ≥ y_min ∧
    y_min = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_value_cubic_equation_l1754_175472


namespace NUMINAMATH_CALUDE_inequality_proof_l1754_175443

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^3 + b^3 + c^3 + (a*b)/(a^2 + b^2) + (b*c)/(b^2 + c^2) + (c*a)/(c^2 + a^2) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1754_175443


namespace NUMINAMATH_CALUDE_total_pears_picked_l1754_175460

def sara_pears : ℕ := 6
def tim_pears : ℕ := 5

theorem total_pears_picked : sara_pears + tim_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l1754_175460


namespace NUMINAMATH_CALUDE_hexagon_count_l1754_175444

theorem hexagon_count (initial_sheets : ℕ) (cuts : ℕ) (initial_sides_per_sheet : ℕ) :
  initial_sheets = 15 →
  cuts = 60 →
  initial_sides_per_sheet = 4 →
  let final_sheets := initial_sheets + cuts
  let total_sides := initial_sheets * initial_sides_per_sheet + cuts * 4
  let hexagon_count := (total_sides - 3 * final_sheets) / 3
  hexagon_count = 25 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_count_l1754_175444


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1754_175481

/-- The number of ways to place 3 different balls into 4 boxes. -/
def total_ways : ℕ := 4^3

/-- The number of ways to place 3 different balls into the first 3 boxes. -/
def ways_without_fourth : ℕ := 3^3

/-- The number of ways to place 3 different balls into 4 boxes,
    such that the 4th box contains at least one ball. -/
def ways_with_fourth : ℕ := total_ways - ways_without_fourth

theorem balls_in_boxes : ways_with_fourth = 37 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1754_175481


namespace NUMINAMATH_CALUDE_rectangle_ordering_l1754_175402

-- Define a rectangle in a Cartesian plane
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

-- Define the "preferable" relation
def preferable (a b : Rectangle) : Prop :=
  (a.x_max ≤ b.x_min) ∨ (a.y_max ≤ b.y_min)

-- Main theorem
theorem rectangle_ordering {n : ℕ} (rectangles : Fin n → Rectangle) 
  (h_nonoverlap : ∀ i j, i ≠ j → 
    (rectangles i).x_max ≤ (rectangles j).x_min ∨
    (rectangles j).x_max ≤ (rectangles i).x_min ∨
    (rectangles i).y_max ≤ (rectangles j).y_min ∨
    (rectangles j).y_max ≤ (rectangles i).y_min) :
  ∃ (σ : Equiv.Perm (Fin n)), ∀ i j, i < j → 
    preferable (rectangles (σ i)) (rectangles (σ j)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ordering_l1754_175402


namespace NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l1754_175466

/-- For a non-zero real number x where x + 1/x is an integer, x^n + 1/x^n is an integer for all natural numbers n. -/
theorem power_sum_reciprocal_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m := by
  sorry

end NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l1754_175466


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1754_175400

theorem largest_angle_in_triangle (x y z : ℝ) : 
  x = 30 ∧ y = 45 ∧ x + y + z = 180 → max x (max y z) = 105 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1754_175400


namespace NUMINAMATH_CALUDE_least_student_number_l1754_175408

theorem least_student_number (p : ℕ) (q : ℕ) : 
  q % 7 = 0 ∧ 
  q ≥ 1000 ∧ 
  q % (p + 1) = 1 ∧ 
  q % (p + 2) = 1 ∧ 
  q % (p + 3) = 1 ∧ 
  (∀ r : ℕ, r % 7 = 0 ∧ 
            r ≥ 1000 ∧ 
            r % (p + 1) = 1 ∧ 
            r % (p + 2) = 1 ∧ 
            r % (p + 3) = 1 → 
            q ≤ r) → 
  q = 1141 :=
by sorry

end NUMINAMATH_CALUDE_least_student_number_l1754_175408


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l1754_175462

/-- Given a train of length 1200 m that crosses a tree in 120 sec,
    prove that it takes 190 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1200
  let tree_crossing_time : ℝ := 120
  let platform_length : ℝ := 700
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  let platform_crossing_time : ℝ := total_distance / train_speed
  platform_crossing_time = 190 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l1754_175462


namespace NUMINAMATH_CALUDE_zoe_bought_eight_roses_l1754_175478

/-- Calculates the number of roses bought given the total spent, cost per flower, and number of daisies. -/
def roses_bought (total_spent : ℕ) (cost_per_flower : ℕ) (num_daisies : ℕ) : ℕ :=
  (total_spent - cost_per_flower * num_daisies) / cost_per_flower

/-- Proves that Zoe bought 8 roses given the problem conditions. -/
theorem zoe_bought_eight_roses (total_spent : ℕ) (cost_per_flower : ℕ) (num_daisies : ℕ) 
    (h1 : total_spent = 30)
    (h2 : cost_per_flower = 3)
    (h3 : num_daisies = 2) : 
  roses_bought total_spent cost_per_flower num_daisies = 8 := by
  sorry

#eval roses_bought 30 3 2  -- Should output 8

end NUMINAMATH_CALUDE_zoe_bought_eight_roses_l1754_175478


namespace NUMINAMATH_CALUDE_projectile_meeting_distance_l1754_175464

theorem projectile_meeting_distance
  (speed1 : ℝ)
  (speed2 : ℝ)
  (meeting_time_minutes : ℝ)
  (h1 : speed1 = 444)
  (h2 : speed2 = 555)
  (h3 : meeting_time_minutes = 120) :
  speed1 * (meeting_time_minutes / 60) + speed2 * (meeting_time_minutes / 60) = 1998 :=
by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_distance_l1754_175464


namespace NUMINAMATH_CALUDE_blue_pigment_percentage_l1754_175486

/-- Represents the composition of the brown paint mixture --/
structure BrownPaint where
  total_weight : ℝ
  blue_percentage : ℝ
  red_weight : ℝ

/-- Represents the composition of the dark blue paint --/
structure DarkBluePaint where
  blue_percentage : ℝ
  red_percentage : ℝ

/-- Represents the composition of the green paint --/
structure GreenPaint where
  blue_percentage : ℝ
  yellow_percentage : ℝ

/-- Theorem stating the percentage of blue pigment in dark blue and green paints --/
theorem blue_pigment_percentage
  (brown : BrownPaint)
  (dark_blue : DarkBluePaint)
  (green : GreenPaint)
  (h1 : brown.total_weight = 10)
  (h2 : brown.blue_percentage = 0.4)
  (h3 : brown.red_weight = 3)
  (h4 : dark_blue.red_percentage = 0.6)
  (h5 : green.yellow_percentage = 0.6)
  (h6 : dark_blue.blue_percentage = green.blue_percentage) :
  dark_blue.blue_percentage = 0.2 :=
sorry


end NUMINAMATH_CALUDE_blue_pigment_percentage_l1754_175486


namespace NUMINAMATH_CALUDE_paths_in_7x7_grid_l1754_175470

/-- The number of paths in a square grid from bottom left to top right -/
def num_paths (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The theorem stating that the number of paths in a 7x7 grid is 3432 -/
theorem paths_in_7x7_grid : num_paths 7 = 3432 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x7_grid_l1754_175470


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l1754_175413

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 560)
  (h2 : floor_width = 240)
  (h3 : tile_length = 60)
  (h4 : tile_width = 56) : 
  (floor_length / tile_length) * (floor_width / tile_width) ≤ 40 ∧ 
  (floor_length / tile_width) * (floor_width / tile_length) ≤ 40 ∧
  ((floor_length / tile_length) * (floor_width / tile_width) = 40 ∨
   (floor_length / tile_width) * (floor_width / tile_length) = 40) :=
by sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l1754_175413


namespace NUMINAMATH_CALUDE_bijective_function_property_l1754_175430

variable {V : Type*} [Fintype V]
variable (f g : V → V)
variable (S T : Set V)

def is_bijective (h : V → V) : Prop :=
  Function.Injective h ∧ Function.Surjective h

theorem bijective_function_property
  (hf : is_bijective f)
  (hg : is_bijective g)
  (hS : S = {w : V | f (f w) = g (g w)})
  (hT : T = {w : V | f (g w) = g (f w)})
  (hST : S ∪ T = Set.univ) :
  ∀ w : V, f w ∈ S ↔ g w ∈ S :=
by sorry

end NUMINAMATH_CALUDE_bijective_function_property_l1754_175430


namespace NUMINAMATH_CALUDE_sum_of_factors_60_l1754_175418

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_60 : sum_of_factors 60 = 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_60_l1754_175418


namespace NUMINAMATH_CALUDE_stormi_bicycle_savings_l1754_175445

/-- The amount of additional money Stormi needs to afford a bicycle -/
def additional_money_needed (num_cars : ℕ) (price_per_car : ℕ) (num_lawns : ℕ) (price_per_lawn : ℕ) (bicycle_cost : ℕ) : ℕ :=
  bicycle_cost - (num_cars * price_per_car + num_lawns * price_per_lawn)

/-- Theorem stating that Stormi needs $24 more to afford the bicycle -/
theorem stormi_bicycle_savings : additional_money_needed 3 10 2 13 80 = 24 := by
  sorry

end NUMINAMATH_CALUDE_stormi_bicycle_savings_l1754_175445


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1754_175431

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in a grid of width w and height h -/
def rectangles_in_grid (w h : ℕ) : ℕ :=
  w * rectangles_in_row h + h * rectangles_in_row w - w * h

theorem rectangles_in_5x4_grid :
  rectangles_in_grid 5 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1754_175431


namespace NUMINAMATH_CALUDE_remainder_equality_l1754_175482

theorem remainder_equality (a b : ℕ) (h1 : a ≠ b) (h2 : a > b) :
  ∃ (q1 q2 r : ℕ), a = (a - b) * q1 + r ∧ b = (a - b) * q2 + r ∧ r < a - b :=
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1754_175482


namespace NUMINAMATH_CALUDE_accurate_estimation_l1754_175403

/-- Represents a scale reading on a measuring device --/
structure ScaleReading where
  min : Float
  max : Float
  reading : Float
  min_le_reading : min ≤ reading
  reading_le_max : reading ≤ max

/-- The most accurate estimation for a scale reading --/
def mostAccurateEstimation (s : ScaleReading) : Float :=
  15.9

/-- Theorem stating that 15.9 is the most accurate estimation for the given scale reading --/
theorem accurate_estimation (s : ScaleReading) 
  (h1 : s.min = 15.75) 
  (h2 : s.max = 16.0) : 
  mostAccurateEstimation s = 15.9 := by
  sorry

end NUMINAMATH_CALUDE_accurate_estimation_l1754_175403


namespace NUMINAMATH_CALUDE_smallest_angle_is_76_l1754_175457

/-- A pentagon with angles in arithmetic sequence -/
structure ArithmeticPentagon where
  -- The common difference between consecutive angles
  d : ℝ
  -- The smallest angle
  a : ℝ
  -- The sum of all angles is 540°
  sum_constraint : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 540
  -- The largest angle is 140°
  max_angle : a + 4*d = 140

/-- 
If a pentagon has angles in arithmetic sequence and its largest angle is 140°,
then its smallest angle is 76°.
-/
theorem smallest_angle_is_76 (p : ArithmeticPentagon) : p.a = 76 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_76_l1754_175457


namespace NUMINAMATH_CALUDE_baseball_team_groups_l1754_175401

/-- The number of groups formed from new and returning players -/
def number_of_groups (new_players returning_players players_per_group : ℕ) : ℕ :=
  (new_players + returning_players) / players_per_group

/-- Theorem stating that the number of groups is 9 given the specific conditions -/
theorem baseball_team_groups : number_of_groups 48 6 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l1754_175401


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1754_175406

theorem complex_sum_theorem (p r s u v x y : ℝ) : 
  let q : ℝ := 4
  let sum_real : ℝ := p + r + u + x
  let sum_imag : ℝ := q + s + v + y
  u = -p - r - x →
  sum_real = 0 →
  sum_imag = 7 →
  s + v + y = 3 := by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1754_175406


namespace NUMINAMATH_CALUDE_hotel_occupancy_and_profit_l1754_175411

/-- Represents a hotel with its pricing and occupancy characteristics -/
structure Hotel where
  totalRooms : ℕ
  originalPrice : ℕ
  fullBookingPrice : ℕ
  costPerRoom : ℕ
  vacancyRate : ℚ
  maxPriceMultiplier : ℚ

/-- Calculates the number of occupied rooms given a price increase -/
def occupiedRooms (h : Hotel) (priceIncrease : ℕ) : ℚ :=
  h.totalRooms - priceIncrease * h.vacancyRate

/-- Calculates the profit given a price increase -/
def profit (h : Hotel) (priceIncrease : ℕ) : ℚ :=
  (h.fullBookingPrice + priceIncrease - h.costPerRoom) * occupiedRooms h priceIncrease

/-- The hotel in the problem -/
def problemHotel : Hotel := {
  totalRooms := 50
  originalPrice := 190
  fullBookingPrice := 180
  costPerRoom := 20
  vacancyRate := 1/10
  maxPriceMultiplier := 3/2
}

theorem hotel_occupancy_and_profit :
  (occupiedRooms problemHotel 50 = 45) ∧
  (profit problemHotel 50 = 9450) := by sorry

end NUMINAMATH_CALUDE_hotel_occupancy_and_profit_l1754_175411


namespace NUMINAMATH_CALUDE_area_inequality_l1754_175492

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A convex quadrilateral with integer vertices -/
structure ConvexQuadrilateral where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint
  convex : Bool  -- Assume this is true for a convex quadrilateral

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : ConvexQuadrilateral) : IntPoint :=
  sorry  -- Definition of diagonal intersection

/-- The area of a shape -/
class HasArea (α : Type) where
  area : α → ℝ

instance : HasArea ConvexQuadrilateral where
  area := sorry  -- Definition of quadrilateral area

instance : HasArea (IntPoint × IntPoint × IntPoint) where
  area := sorry  -- Definition of triangle area

theorem area_inequality (q : ConvexQuadrilateral) :
  let S := diagonalIntersection q
  let P := HasArea.area q
  let P₁ := HasArea.area (q.A, q.B, S)
  Real.sqrt P ≥ Real.sqrt P₁ + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_inequality_l1754_175492


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l1754_175429

theorem absolute_value_sum_difference (x y : ℚ) 
  (hx : |x| = 9) (hy : |y| = 5) : 
  ((x < 0 ∧ y > 0) → x + y = -4) ∧
  (|x + y| = x + y → (x - y = 4 ∨ x - y = 14)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l1754_175429


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1754_175467

theorem inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → 4^x - 2^(x+1) - a ≤ 0) →
  a ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1754_175467


namespace NUMINAMATH_CALUDE_analogical_reasoning_example_l1754_175484

/-- Represents different types of reasoning -/
inductive ReasoningType
  | Deductive
  | Inductive
  | Analogical
  | Other

/-- Determines the type of reasoning for a given statement -/
def determineReasoningType (statement : String) : ReasoningType :=
  match statement with
  | "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" => ReasoningType.Analogical
  | _ => ReasoningType.Other

/-- Theorem stating that the given statement is an example of analogical reasoning -/
theorem analogical_reasoning_example :
  determineReasoningType "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" = ReasoningType.Analogical := by
  sorry


end NUMINAMATH_CALUDE_analogical_reasoning_example_l1754_175484


namespace NUMINAMATH_CALUDE_min_value_inequality_l1754_175407

theorem min_value_inequality (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1754_175407


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1754_175477

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := c / a
  (a^2 * b^2 = (a^2 - c^2) * b^2) →  -- Ellipse equation
  (c^2 + b^2 = a^2) →               -- Right triangle condition
  e = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1754_175477


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l1754_175476

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters (total pink blue : ℕ) : ℕ := total - pink - blue

/-- Theorem stating the number of yellow highlighters -/
theorem yellow_highlighters_count :
  yellow_highlighters 22 9 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l1754_175476


namespace NUMINAMATH_CALUDE_least_odd_number_satisfying_conditions_l1754_175463

theorem least_odd_number_satisfying_conditions : ∃ (m₁ m₂ n₁ n₂ : ℕ+), 
  let a : ℕ := 261
  (a = m₁.val ^ 2 + n₁.val ^ 2) ∧
  (a ^ 2 = m₂.val ^ 2 + n₂.val ^ 2) ∧
  (m₁.val - n₁.val = m₂.val - n₂.val) ∧
  (∀ (b : ℕ) (k₁ k₂ l₁ l₂ : ℕ+), b < a → b % 2 = 1 → b > 5 →
    (b = k₁.val ^ 2 + l₁.val ^ 2 ∧
     b ^ 2 = k₂.val ^ 2 + l₂.val ^ 2 ∧
     k₁.val - l₁.val = k₂.val - l₂.val) → False) :=
by sorry

end NUMINAMATH_CALUDE_least_odd_number_satisfying_conditions_l1754_175463


namespace NUMINAMATH_CALUDE_secret_spread_l1754_175491

/-- Represents the number of people each person tells the secret to on a given day -/
def tell_count (day : Nat) : Nat :=
  match day with
  | 1 => 1  -- Monday: Jessica tells 1 friend
  | 2 => 2  -- Tuesday
  | 3 => 2  -- Wednesday
  | 4 => 1  -- Thursday
  | _ => 2  -- Friday to Monday

/-- Calculates the total number of people knowing the secret after a given number of days -/
def total_knowing (days : Nat) : Nat :=
  match days with
  | 0 => 1  -- Only Jessica knows on day 0
  | n + 1 => total_knowing n + (total_knowing n - total_knowing (n - 1)) * tell_count (n + 1)

/-- The theorem stating that after 8 days, 132 people will know the secret -/
theorem secret_spread : total_knowing 8 = 132 := by
  sorry


end NUMINAMATH_CALUDE_secret_spread_l1754_175491


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1754_175441

theorem geometric_sequence_sum (a b c q : ℝ) (h_seq : (a + b + c) * q = b + c - a ∧
                                                    (b + c - a) * q = c + a - b ∧
                                                    (c + a - b) * q = a + b - c) :
  q^3 + q^2 + q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1754_175441


namespace NUMINAMATH_CALUDE_total_trophies_is_430_l1754_175453

/-- Calculates the total number of trophies Jack and Michael will have after three years -/
def totalTrophiesAfterThreeYears (michaelCurrentTrophies : ℕ) (michaelTrophyIncrease : ℕ) (jackMultiplier : ℕ) : ℕ :=
  let michaelFutureTrophies := michaelCurrentTrophies + michaelTrophyIncrease
  let jackFutureTrophies := jackMultiplier * michaelCurrentTrophies
  michaelFutureTrophies + jackFutureTrophies

/-- Theorem stating that the total number of trophies after three years is 430 -/
theorem total_trophies_is_430 : 
  totalTrophiesAfterThreeYears 30 100 10 = 430 := by
  sorry

end NUMINAMATH_CALUDE_total_trophies_is_430_l1754_175453


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1754_175440

/-- The number of ways to seat 2 students in a row of 5 desks with at least one empty desk between them -/
def seatingArrangements : ℕ := 6

/-- The number of desks in the row -/
def numDesks : ℕ := 5

/-- The number of students to be seated -/
def numStudents : ℕ := 2

/-- The minimum number of empty desks required between the students -/
def minEmptyDesks : ℕ := 1

theorem correct_seating_arrangements :
  seatingArrangements = 
    (numDesks - numStudents - minEmptyDesks + 1) * (numStudents) :=
by sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1754_175440


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1754_175432

theorem smallest_n_square_and_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 ∧ (∃ (y : ℕ), 4 * x = y^2) ∧ (∃ (z : ℕ), 3 * x = z^3) → x ≥ n) ∧
  n = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1754_175432


namespace NUMINAMATH_CALUDE_satellite_has_24_units_l1754_175416

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  total_upgraded : ℕ

/-- The conditions of the satellite problem. -/
def satellite_conditions (s : Satellite) : Prop :=
  -- Condition 2: non-upgraded sensors per unit is 1/6 of total upgraded
  s.non_upgraded_per_unit = s.total_upgraded / 6 ∧
  -- Condition 3: 20% of all sensors are upgraded
  s.total_upgraded = (s.total_upgraded + s.units * s.non_upgraded_per_unit) / 5

/-- The theorem stating that a satellite satisfying the given conditions has 24 units. -/
theorem satellite_has_24_units (s : Satellite) (h : satellite_conditions s) : s.units = 24 := by
  sorry


end NUMINAMATH_CALUDE_satellite_has_24_units_l1754_175416


namespace NUMINAMATH_CALUDE_ten_thousands_representation_l1754_175426

def ten_thousands : ℕ := 10000

def three_thousand_nine_hundred_seventy_six : ℕ := 3976

theorem ten_thousands_representation :
  three_thousand_nine_hundred_seventy_six * ten_thousands = 39760000 ∧
  three_thousand_nine_hundred_seventy_six = 3976 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousands_representation_l1754_175426


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_y_coord_l1754_175450

/-- The y-coordinate of the intersection point of perpendicular tangents on a parabola -/
theorem perpendicular_tangents_intersection_y_coord 
  (a b : ℝ) 
  (h_parabola : ∀ x : ℝ, (x = a ∨ x = b) → 4 * x^2 = (4 * x^2))
  (h_perpendicular : (8 * a) * (8 * b) = -1) :
  ∃ P : ℝ × ℝ, 
    (P.1 = (a + b) / 2) ∧ 
    (P.2 = 4 * a * b) ∧ 
    (P.2 = -1/8) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_y_coord_l1754_175450


namespace NUMINAMATH_CALUDE_cube_plus_n_minus_two_power_of_two_l1754_175410

theorem cube_plus_n_minus_two_power_of_two (n : ℕ+) :
  (∃ k : ℕ, (n : ℕ)^3 + n - 2 = 2^k) ↔ n = 2 ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_n_minus_two_power_of_two_l1754_175410


namespace NUMINAMATH_CALUDE_light_path_in_cube_l1754_175487

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light beam in the cube -/
structure LightBeam where
  start : Point3D
  reflectionPoint : Point3D

/-- The length of the light path in the cube -/
def lightPathLength (c : Cube) (lb : LightBeam) : ℝ :=
  sorry

theorem light_path_in_cube (c : Cube) (lb : LightBeam) :
  c.sideLength = 10 ∧
  lb.start = Point3D.mk 0 0 0 ∧
  lb.reflectionPoint = Point3D.mk 6 4 10 →
  lightPathLength c lb = 10 * Real.sqrt 152 :=
sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l1754_175487


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1754_175483

theorem sum_of_solutions_quadratic (a b c d e : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let g : ℝ → ℝ := λ x => d * x + e
  (∀ x, f x = g x) →
  (-(b - d) / (2 * a)) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1754_175483


namespace NUMINAMATH_CALUDE_pond_fish_problem_l1754_175435

/-- Represents the number of fish in a pond -/
def total_fish : ℕ := 500

/-- Represents the number of fish initially tagged -/
def tagged_fish : ℕ := 50

/-- Represents the number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- Represents the number of tagged fish found in the second catch -/
def tagged_in_second_catch : ℕ := 5

theorem pond_fish_problem :
  (tagged_in_second_catch : ℚ) / second_catch = tagged_fish / total_fish :=
sorry

end NUMINAMATH_CALUDE_pond_fish_problem_l1754_175435


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l1754_175479

theorem geometric_progression_ratio (a b c d x y z r : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  a * x * (y - z) ≠ 0 →
  b * y * (z - x) ≠ 0 →
  c * z * (x - y) ≠ 0 →
  d * x * (y - z) ≠ 0 →
  a * x * (y - z) ≠ b * y * (z - x) →
  b * y * (z - x) ≠ c * z * (x - y) →
  c * z * (x - y) ≠ d * x * (y - z) →
  (∃ k : ℝ, k ≠ 0 ∧ 
    b * y * (z - x) = k * (a * x * (y - z)) ∧
    c * z * (x - y) = k * (b * y * (z - x)) ∧
    d * x * (y - z) = k * (c * z * (x - y))) →
  r^3 + r^2 + r + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l1754_175479


namespace NUMINAMATH_CALUDE_brick_width_proof_l1754_175442

/-- The width of a brick that satisfies the given conditions --/
def brick_width : ℝ := 11.25

theorem brick_width_proof (wall_volume : ℝ) (brick_length : ℝ) (brick_height : ℝ) (num_bricks : ℕ) 
  (h1 : wall_volume = 800 * 600 * 22.5)
  (h2 : ∀ w, brick_length * w * brick_height * num_bricks = wall_volume)
  (h3 : brick_length = 50)
  (h4 : brick_height = 6)
  (h5 : num_bricks = 3200) :
  brick_width = 11.25 := by
sorry

end NUMINAMATH_CALUDE_brick_width_proof_l1754_175442


namespace NUMINAMATH_CALUDE_water_formed_equals_three_l1754_175433

-- Define the chemical species
inductive ChemicalSpecies
| NH4Cl
| NaOH
| NaCl
| NH3
| H2O

-- Define the reaction equation
def reactionEquation : List (ChemicalSpecies × Int) :=
  [(ChemicalSpecies.NH4Cl, -1), (ChemicalSpecies.NaOH, -1),
   (ChemicalSpecies.NaCl, 1), (ChemicalSpecies.NH3, 1), (ChemicalSpecies.H2O, 1)]

-- Define the initial amounts of reactants
def initialNH4Cl : ℕ := 3
def initialNaOH : ℕ := 3

-- Function to calculate the moles of water formed
def molesOfWaterFormed (nh4cl : ℕ) (naoh : ℕ) : ℕ :=
  min nh4cl naoh

-- Theorem statement
theorem water_formed_equals_three :
  molesOfWaterFormed initialNH4Cl initialNaOH = 3 := by
  sorry


end NUMINAMATH_CALUDE_water_formed_equals_three_l1754_175433


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l1754_175414

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | f x > 1}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (2/3) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l1754_175414


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1754_175465

theorem sine_cosine_inequality (α β : ℝ) (h : 0 < α + β ∧ α + β ≤ π) :
  (Real.sin α - Real.sin β) * (Real.cos α - Real.cos β) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1754_175465


namespace NUMINAMATH_CALUDE_town_population_problem_l1754_175427

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1500) * 85 / 100) : ℕ) = original_population - 45 →
  original_population = 8800 := by
sorry

end NUMINAMATH_CALUDE_town_population_problem_l1754_175427


namespace NUMINAMATH_CALUDE_range_of_m_l1754_175447

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|1 - (x-1)/2| ≤ 3 → x^2 - 2*x + 1 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 ∧ |1 - (x-1)/2| > 3)) ∧ 
  m > 0 → 
  m ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1754_175447


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l1754_175452

theorem smallest_positive_integer_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 3 ∧ 
  x % 8 = 5 ∧
  (∀ y : ℕ, y > 0 ∧ y % 6 = 3 ∧ y % 8 = 5 → x ≤ y) ∧
  x = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l1754_175452


namespace NUMINAMATH_CALUDE_x_equals_nine_l1754_175423

/-- The star operation defined as a ⭐ b = 5a - 3b -/
def star (a b : ℝ) : ℝ := 5 * a - 3 * b

/-- Theorem stating that X = 9 given the condition X ⭐ (3 ⭐ 2) = 18 -/
theorem x_equals_nine : ∃ X : ℝ, star X (star 3 2) = 18 ∧ X = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_nine_l1754_175423


namespace NUMINAMATH_CALUDE_power_calculation_l1754_175421

theorem power_calculation : ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1754_175421


namespace NUMINAMATH_CALUDE_largest_n_for_inequality_l1754_175415

theorem largest_n_for_inequality : ∃ (n : ℕ), n = 24 ∧ 
  (∀ (a b c d : ℝ), 
    (↑n + 2) * Real.sqrt (a^2 + b^2) + 
    (↑n + 1) * Real.sqrt (a^2 + c^2) + 
    (↑n + 1) * Real.sqrt (a^2 + d^2) ≥ 
    ↑n * (a + b + c + d)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (a b c d : ℝ), 
      (↑m + 2) * Real.sqrt (a^2 + b^2) + 
      (↑m + 1) * Real.sqrt (a^2 + c^2) + 
      (↑m + 1) * Real.sqrt (a^2 + d^2) < 
      ↑m * (a + b + c + d)) :=
by sorry


end NUMINAMATH_CALUDE_largest_n_for_inequality_l1754_175415


namespace NUMINAMATH_CALUDE_badge_exchange_l1754_175485

theorem badge_exchange (x : ℕ) : 
  (x + 5 - (6 * (x + 5)) / 25 + x / 5 = x - x / 5 + (6 * (x + 5)) / 25 - 1) → 
  (x = 45 ∧ x + 5 = 50) := by
  sorry

end NUMINAMATH_CALUDE_badge_exchange_l1754_175485


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1754_175495

theorem division_remainder_problem (k : ℕ+) (h : ∃ b : ℕ, 80 = b * k^2 + 8) :
  ∃ q : ℕ, 140 = q * k + 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1754_175495


namespace NUMINAMATH_CALUDE_leo_balloon_distribution_l1754_175404

theorem leo_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 144) 
  (h2 : num_friends = 9) :
  total_balloons % num_friends = 0 := by
  sorry

end NUMINAMATH_CALUDE_leo_balloon_distribution_l1754_175404


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l1754_175471

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, -2], ![1, 1]]) : 
  (A^2)⁻¹ = ![![7, -8], ![4, -1]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l1754_175471


namespace NUMINAMATH_CALUDE_rectangle_lcm_gcd_product_l1754_175424

theorem rectangle_lcm_gcd_product : 
  let a : ℕ := 24
  let b : ℕ := 36
  Nat.lcm a b * Nat.gcd a b = 864 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_lcm_gcd_product_l1754_175424


namespace NUMINAMATH_CALUDE_remainder_eight_pow_six_plus_one_mod_seven_l1754_175409

theorem remainder_eight_pow_six_plus_one_mod_seven :
  (8^6 + 1) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_pow_six_plus_one_mod_seven_l1754_175409


namespace NUMINAMATH_CALUDE_balance_condition1_balance_condition2_triangular_weight_is_60_l1754_175489

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- First balance condition: 1 round + 1 triangular = 3 round -/
theorem balance_condition1 : round_weight + triangular_weight = 3 * round_weight := by sorry

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
theorem balance_condition2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight := by sorry

/-- Proof that the triangular weight is 60 grams -/
theorem triangular_weight_is_60 : triangular_weight = 60 := by sorry

end NUMINAMATH_CALUDE_balance_condition1_balance_condition2_triangular_weight_is_60_l1754_175489


namespace NUMINAMATH_CALUDE_unique_m_for_direct_proportion_l1754_175474

/-- A function f(x) is a direct proportion function if it can be written as f(x) = kx for some non-zero constant k. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m+1)x + m^2 - 1 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 1) * x + m^2 - 1

/-- Theorem: The only value of m that makes f(m) a direct proportion function is 1 -/
theorem unique_m_for_direct_proportion :
  ∃! m : ℝ, IsDirectProportion (f m) ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_for_direct_proportion_l1754_175474


namespace NUMINAMATH_CALUDE_log_xy_value_l1754_175461

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^5) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x * y) = 6/7 := by sorry

end NUMINAMATH_CALUDE_log_xy_value_l1754_175461


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1754_175405

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  1/a + 1/b ≥ 3 + 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1754_175405


namespace NUMINAMATH_CALUDE_extreme_values_and_increasing_condition_l1754_175422

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem extreme_values_and_increasing_condition :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → f (-1/2) x ≤ f (-1/2) x₀) ∧
  f (-1/2) x₀ = 0 ∧
  (∀ (y : ℝ), y > 0 → ∃ (z : ℝ), z > y ∧ f (-1/2) z > f (-1/2) y) ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), 0 < x ∧ x < y → f a x < f a y) ↔ a ≥ 1 / (2 * Real.exp 2)) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_increasing_condition_l1754_175422


namespace NUMINAMATH_CALUDE_binomial_cube_example_l1754_175469

theorem binomial_cube_example : 4^3 + 3*(4^2)*2 + 3*4*(2^2) + 2^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_example_l1754_175469
