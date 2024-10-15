import Mathlib

namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l639_63919

theorem shaded_area_between_circles (r₁ r₂ chord_length : ℝ) 
  (h₁ : r₁ = 40)
  (h₂ : r₂ = 60)
  (h₃ : chord_length = 100)
  (h₄ : r₁ < r₂)
  (h₅ : chord_length^2 = 4 * (r₂^2 - r₁^2)) : -- Condition for tangency
  (π * r₂^2 - π * r₁^2) = 2000 * π :=
sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l639_63919


namespace NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l639_63955

theorem three_numbers_in_unit_interval (x y z : ℝ) :
  (0 ≤ x ∧ x < 1) → (0 ≤ y ∧ y < 1) → (0 ≤ z ∧ z < 1) →
  ∃ a b : ℝ, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l639_63955


namespace NUMINAMATH_CALUDE_seven_thirteenths_of_3940_percent_of_25000_l639_63937

theorem seven_thirteenths_of_3940_percent_of_25000 : 
  (7 / 13 * 3940) / 25000 * 100 = 8.484 := by
  sorry

end NUMINAMATH_CALUDE_seven_thirteenths_of_3940_percent_of_25000_l639_63937


namespace NUMINAMATH_CALUDE_tom_seashells_per_day_l639_63989

/-- Represents the number of seashells Tom found each day -/
def seashells_per_day (total_seashells : ℕ) (days_at_beach : ℕ) : ℕ :=
  total_seashells / days_at_beach

/-- Theorem stating that Tom found 7 seashells per day -/
theorem tom_seashells_per_day :
  seashells_per_day 35 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_per_day_l639_63989


namespace NUMINAMATH_CALUDE_expression_evaluation_l639_63999

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^2)^y * (y^3)^x / ((y^2)^y * (x^3)^x) = x^(2*y - 3*x) * y^(3*x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l639_63999


namespace NUMINAMATH_CALUDE_sector_angle_l639_63992

/-- Given a circular sector with circumference 8 and area 4, 
    prove that the central angle in radians is 2. -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_circumference : α * r + 2 * r = 8) 
  (h_area : (1 / 2) * α * r^2 = 4) : 
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l639_63992


namespace NUMINAMATH_CALUDE_cricket_players_count_l639_63925

/-- The number of cricket players in a games hour -/
def cricket_players (total_players hockey_players football_players softball_players : ℕ) : ℕ :=
  total_players - (hockey_players + football_players + softball_players)

/-- Theorem: There are 16 cricket players given the conditions -/
theorem cricket_players_count : cricket_players 59 12 18 13 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l639_63925


namespace NUMINAMATH_CALUDE_parabola_translation_l639_63931

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c - v }

theorem parabola_translation (p : Parabola) :
  p.a = 1/2 ∧ p.b = 0 ∧ p.c = 1 →
  let p' := translate p 1 3
  p'.a = 1/2 ∧ p'.b = 1 ∧ p'.c = -3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l639_63931


namespace NUMINAMATH_CALUDE_stamp_collection_l639_63998

theorem stamp_collection (aj kj cj bj : ℕ) : 
  kj = aj / 2 →
  cj = 2 * kj + 5 →
  bj = 3 * aj - 3 →
  aj + kj + cj + bj = 1472 →
  aj = 267 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_l639_63998


namespace NUMINAMATH_CALUDE_max_symmetry_axes_is_2k_l639_63910

/-- The maximum number of axes of symmetry for the union of k line segments on a plane -/
def max_symmetry_axes (k : ℕ) : ℕ := 2 * k

/-- Theorem: The maximum number of axes of symmetry for the union of k line segments on a plane is 2k -/
theorem max_symmetry_axes_is_2k (k : ℕ) :
  max_symmetry_axes k = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_max_symmetry_axes_is_2k_l639_63910


namespace NUMINAMATH_CALUDE_circle_C_equation_l639_63906

/-- A circle C with center on the x-axis passing through points A(-1,1) and B(1,3) -/
structure CircleC where
  center : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  passes_through_A : (center.1 + 1)^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + (center.2 - 3)^2

/-- The equation of circle C is (x-2)²+y²=10 -/
theorem circle_C_equation (C : CircleC) :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 10 ↔ (x - C.center.1)^2 + (y - C.center.2)^2 = (C.center.1 + 1)^2 + (C.center.2 - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_C_equation_l639_63906


namespace NUMINAMATH_CALUDE_sequence_max_value_l639_63981

theorem sequence_max_value (n : ℤ) : -n^2 + 15*n + 3 ≤ 59 := by
  sorry

end NUMINAMATH_CALUDE_sequence_max_value_l639_63981


namespace NUMINAMATH_CALUDE_total_teachers_is_182_l639_63921

/-- Represents the number of teachers in different categories and sampling information -/
structure TeacherInfo where
  senior_teachers : ℕ
  intermediate_teachers : ℕ
  total_sampled : ℕ
  other_sampled : ℕ

/-- Calculates the total number of teachers in the school -/
def total_teachers (info : TeacherInfo) : ℕ :=
  let senior_intermediate := info.senior_teachers + info.intermediate_teachers
  let senior_intermediate_sampled := info.total_sampled - info.other_sampled
  (senior_intermediate * info.total_sampled) / senior_intermediate_sampled

/-- Theorem stating that the total number of teachers is 182 -/
theorem total_teachers_is_182 (info : TeacherInfo) 
    (h1 : info.senior_teachers = 26)
    (h2 : info.intermediate_teachers = 104)
    (h3 : info.total_sampled = 56)
    (h4 : info.other_sampled = 16) :
    total_teachers info = 182 := by
  sorry

#eval total_teachers { senior_teachers := 26, intermediate_teachers := 104, total_sampled := 56, other_sampled := 16 }

end NUMINAMATH_CALUDE_total_teachers_is_182_l639_63921


namespace NUMINAMATH_CALUDE_square_root_of_49_l639_63991

theorem square_root_of_49 : (Real.sqrt 49)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_49_l639_63991


namespace NUMINAMATH_CALUDE_angle_ratios_l639_63927

theorem angle_ratios (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 ∧
  2 + Real.sin α * Real.cos α - (Real.cos α)^2 = 22/25 := by
  sorry

end NUMINAMATH_CALUDE_angle_ratios_l639_63927


namespace NUMINAMATH_CALUDE_square_free_divisibility_l639_63916

theorem square_free_divisibility (n : ℕ) (h1 : n > 1) (h2 : Squarefree n) :
  ∃ (p m : ℕ), Prime p ∧ p ∣ n ∧ n ∣ p^2 + p * m^p :=
sorry

end NUMINAMATH_CALUDE_square_free_divisibility_l639_63916


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_box_l639_63977

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

theorem min_cubes_for_specific_box :
  min_cubes_for_box 7 18 3 9 = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_specific_box_l639_63977


namespace NUMINAMATH_CALUDE_apartment_fractions_l639_63956

theorem apartment_fractions (one_bedroom : Real) (two_bedroom : Real) 
  (h1 : one_bedroom = 0.17)
  (h2 : one_bedroom + two_bedroom = 0.5) :
  two_bedroom = 0.33 := by
sorry

end NUMINAMATH_CALUDE_apartment_fractions_l639_63956


namespace NUMINAMATH_CALUDE_nancy_mexican_antacids_l639_63901

/-- Represents the number of antacids Nancy takes per day when eating Mexican food -/
def mexican_antacids : ℕ := sorry

/-- Represents the number of antacids Nancy takes per day when eating Indian food -/
def indian_antacids : ℕ := 3

/-- Represents the number of antacids Nancy takes per day when eating other food -/
def other_antacids : ℕ := 1

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_meals_per_week : ℕ := 3

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_meals_per_week : ℕ := 2

/-- Represents the number of antacids Nancy takes per month -/
def antacids_per_month : ℕ := 60

/-- Represents the number of weeks in a month (approximated) -/
def weeks_per_month : ℕ := 4

theorem nancy_mexican_antacids : 
  mexican_antacids = 2 :=
by sorry

end NUMINAMATH_CALUDE_nancy_mexican_antacids_l639_63901


namespace NUMINAMATH_CALUDE_bills_ratio_l639_63947

/-- Proves that the ratio of bills Geric had to bills Kyla had at the beginning is 2:1 --/
theorem bills_ratio (jessa_bills_after geric_bills kyla_bills : ℕ) : 
  jessa_bills_after = 7 →
  geric_bills = 16 →
  kyla_bills = (jessa_bills_after + 3) - 2 →
  (geric_bills : ℚ) / kyla_bills = 2 := by
  sorry

end NUMINAMATH_CALUDE_bills_ratio_l639_63947


namespace NUMINAMATH_CALUDE_max_product_on_curve_l639_63935

theorem max_product_on_curve (x y : ℝ) :
  0 ≤ x ∧ x ≤ 12 ∧ 0 ≤ y ∧ y ≤ 12 →
  x * y = (12 - x)^2 * (12 - y)^2 →
  x * y ≤ 81 :=
by sorry

end NUMINAMATH_CALUDE_max_product_on_curve_l639_63935


namespace NUMINAMATH_CALUDE_train_arrival_interval_l639_63957

def minutes_between (h1 m1 h2 m2 : ℕ) : ℕ :=
  (h2 * 60 + m2) - (h1 * 60 + m1)

theorem train_arrival_interval (x : ℕ) : 
  x > 0 → 
  minutes_between 10 10 10 55 % x = 0 → 
  minutes_between 10 55 11 58 % x = 0 → 
  x = 9 :=
sorry

end NUMINAMATH_CALUDE_train_arrival_interval_l639_63957


namespace NUMINAMATH_CALUDE_sum_of_five_numbers_l639_63972

theorem sum_of_five_numbers : 1357 + 2468 + 3579 + 4680 + 5791 = 17875 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_numbers_l639_63972


namespace NUMINAMATH_CALUDE_std_dev_and_range_invariance_l639_63905

variable {n : ℕ} (c : ℝ)
variable (X Y : Fin n → ℝ)

def add_constant (X : Fin n → ℝ) (c : ℝ) : Fin n → ℝ :=
  fun i => X i + c

def sample_std_dev (X : Fin n → ℝ) : ℝ := sorry

def sample_range (X : Fin n → ℝ) : ℝ := sorry

theorem std_dev_and_range_invariance
  (h_nonzero : c ≠ 0)
  (h_Y : Y = add_constant X c) :
  sample_std_dev X = sample_std_dev Y ∧
  sample_range X = sample_range Y := by sorry

end NUMINAMATH_CALUDE_std_dev_and_range_invariance_l639_63905


namespace NUMINAMATH_CALUDE_mo_negative_bo_positive_l639_63953

-- Define the two types of people
inductive PersonType
| Positive
| Negative

-- Define a person with a type
structure Person where
  name : String
  type : PersonType

-- Define the property of asking a question
def asksQuestion (p : Person) (q : Prop) : Prop :=
  match p.type with
  | PersonType.Positive => q
  | PersonType.Negative => ¬q

-- Define Mo and Bo
def Mo : Person := { name := "Mo", type := PersonType.Negative }
def Bo : Person := { name := "Bo", type := PersonType.Positive }

-- Define the question Mo asked
def moQuestion : Prop := Mo.type = PersonType.Negative ∧ Bo.type = PersonType.Negative

-- Theorem stating that Mo is negative and Bo is positive
theorem mo_negative_bo_positive :
  asksQuestion Mo moQuestion ∧ (Mo.type = PersonType.Negative ∧ Bo.type = PersonType.Positive) :=
by sorry


end NUMINAMATH_CALUDE_mo_negative_bo_positive_l639_63953


namespace NUMINAMATH_CALUDE_twelve_percent_greater_than_80_l639_63950

theorem twelve_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 12 / 100) → x = 89.6 := by
sorry

end NUMINAMATH_CALUDE_twelve_percent_greater_than_80_l639_63950


namespace NUMINAMATH_CALUDE_employment_percentage_l639_63900

theorem employment_percentage (total_population : ℝ) 
  (employed_males_percentage : ℝ) (employed_females_percentage : ℝ) :
  employed_males_percentage = 48 →
  employed_females_percentage = 20 →
  (employed_males_percentage / (100 - employed_females_percentage)) * 100 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_percentage_l639_63900


namespace NUMINAMATH_CALUDE_gasoline_price_change_l639_63958

/-- Represents the price change of gasoline over two months -/
theorem gasoline_price_change 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (x : ℝ) 
  (h1 : initial_price = 7.5)
  (h2 : final_price = 8.4)
  (h3 : x ≥ 0) -- Assuming non-negative growth rate
  : initial_price * (1 + x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_gasoline_price_change_l639_63958


namespace NUMINAMATH_CALUDE_expand_expression_l639_63959

theorem expand_expression (x y : ℝ) : (16*x + 18 - 7*y) * 3*x = 48*x^2 + 54*x - 21*x*y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l639_63959


namespace NUMINAMATH_CALUDE_product_sum_inequality_l639_63974

theorem product_sum_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l639_63974


namespace NUMINAMATH_CALUDE_restaurant_order_combinations_l639_63975

theorem restaurant_order_combinations :
  let main_dish_options : ℕ := 12
  let side_dish_options : ℕ := 5
  let person_count : ℕ := 2
  main_dish_options ^ person_count * side_dish_options = 720 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_order_combinations_l639_63975


namespace NUMINAMATH_CALUDE_catchup_time_correct_l639_63971

/-- The time (in hours) it takes for the second car to catch up with the first car -/
def catchup_time : ℝ := 1.5

/-- The speed of the first car in km/h -/
def speed1 : ℝ := 60

/-- The speed of the second car in km/h -/
def speed2 : ℝ := 80

/-- The head start time of the first car in hours -/
def head_start : ℝ := 0.5

/-- Theorem stating that the catchup time is correct given the conditions -/
theorem catchup_time_correct : 
  speed1 * (catchup_time + head_start) = speed2 * catchup_time := by
  sorry

#check catchup_time_correct

end NUMINAMATH_CALUDE_catchup_time_correct_l639_63971


namespace NUMINAMATH_CALUDE_average_difference_l639_63924

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (20 + 60 + x) / 3 + 5 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l639_63924


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l639_63996

theorem probability_nine_heads_in_twelve_flips :
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1-p)^(n-k) = 220/4096 := by
sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l639_63996


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_10080_l639_63990

def digit_product (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_10080 :
  ∀ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ digit_product n = 10080 → n ≤ 98754 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_10080_l639_63990


namespace NUMINAMATH_CALUDE_system_no_solution_l639_63943

theorem system_no_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y = 1 ∧ 2 * x + a * y = 1) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l639_63943


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l639_63988

/-- Represents a number in a given base with two identical digits --/
def twoDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid in a given base --/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (C D : Nat),
    isValidDigit C 4 ∧
    isValidDigit D 6 ∧
    twoDigitNumber C 4 = 35 ∧
    twoDigitNumber D 6 = 35 ∧
    (∀ (C' D' : Nat),
      isValidDigit C' 4 →
      isValidDigit D' 6 →
      twoDigitNumber C' 4 = twoDigitNumber D' 6 →
      twoDigitNumber C' 4 ≥ 35) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l639_63988


namespace NUMINAMATH_CALUDE_cosine_function_theorem_l639_63909

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem cosine_function_theorem (f : ℝ → ℝ) (T : ℝ) :
  is_periodic f T →
  (∀ x, Real.cos x = f x - 2 * f (x - π)) →
  (∀ x, Real.cos x = f (x - T) - 2 * f (x - T - π)) →
  (∀ x, Real.cos x = Real.cos (x - T)) →
  (∀ x, f x = (1/3) * Real.cos x) :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_theorem_l639_63909


namespace NUMINAMATH_CALUDE_distribute_five_to_two_nonempty_l639_63907

theorem distribute_five_to_two_nonempty (n : Nat) (k : Nat) : 
  n = 5 → k = 2 → (Finset.sum (Finset.range (n - 1)) (λ i => Nat.choose n (i + 1) * 2)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_two_nonempty_l639_63907


namespace NUMINAMATH_CALUDE_parabola_a_range_l639_63963

/-- A parabola that opens downwards -/
structure DownwardParabola where
  a : ℝ
  eq : ℝ → ℝ := λ x => a * x^2 - 2 * a * x + 3
  opens_downward : a < 0

/-- The theorem stating the range of 'a' for a downward parabola with positive y-values in (0, 3) -/
theorem parabola_a_range (p : DownwardParabola) 
  (h : ∀ x, 0 < x → x < 3 → p.eq x > 0) : 
  -1 < p.a ∧ p.a < 0 := by
  sorry


end NUMINAMATH_CALUDE_parabola_a_range_l639_63963


namespace NUMINAMATH_CALUDE_chipmunk_families_went_away_l639_63961

theorem chipmunk_families_went_away (original : ℕ) (left : ℕ) (h1 : original = 86) (h2 : left = 21) :
  original - left = 65 := by
  sorry

end NUMINAMATH_CALUDE_chipmunk_families_went_away_l639_63961


namespace NUMINAMATH_CALUDE_point_on_line_l639_63960

/-- A line in the xy-plane defined by two points -/
structure Line where
  x1 : ℚ
  y1 : ℚ
  x2 : ℚ
  y2 : ℚ

/-- Check if a point (x, y) lies on the given line -/
def Line.contains (l : Line) (x y : ℚ) : Prop :=
  (y - l.y1) * (l.x2 - l.x1) = (x - l.x1) * (l.y2 - l.y1)

theorem point_on_line (l : Line) (x : ℚ) :
  l.x1 = 1 ∧ l.y1 = 9 ∧ l.x2 = -2 ∧ l.y2 = -1 →
  l.contains x 2 →
  x = -11/10 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l639_63960


namespace NUMINAMATH_CALUDE_product_of_larger_numbers_l639_63997

theorem product_of_larger_numbers (A B C : ℝ) 
  (h1 : B - A = C - B) 
  (h2 : A * B = 85) 
  (h3 : B = 10) : 
  B * C = 115 := by
sorry

end NUMINAMATH_CALUDE_product_of_larger_numbers_l639_63997


namespace NUMINAMATH_CALUDE_cost_of_pens_calculation_l639_63946

/-- The cost of the box of pens Linda bought -/
def cost_of_pens : ℝ := 1.70

/-- The number of notebooks Linda bought -/
def num_notebooks : ℕ := 3

/-- The cost of each notebook -/
def cost_per_notebook : ℝ := 1.20

/-- The cost of the box of pencils -/
def cost_of_pencils : ℝ := 1.50

/-- The total amount Linda spent -/
def total_spent : ℝ := 6.80

theorem cost_of_pens_calculation :
  cost_of_pens = total_spent - (↑num_notebooks * cost_per_notebook + cost_of_pencils) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_pens_calculation_l639_63946


namespace NUMINAMATH_CALUDE_fermat_fourth_power_l639_63973

theorem fermat_fourth_power (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^4 + y^4 ≠ z^4 := by
  sorry

end NUMINAMATH_CALUDE_fermat_fourth_power_l639_63973


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l639_63939

/-- Elaine's earnings last year -/
def last_year_earnings : ℝ := 1

/-- Percentage of earnings spent on rent last year -/
def last_year_rent_percentage : ℝ := 20

/-- Earnings increase percentage this year -/
def earnings_increase : ℝ := 20

/-- Percentage of earnings spent on rent this year -/
def this_year_rent_percentage : ℝ := 30

/-- Increase in rent amount from last year to this year -/
def rent_increase : ℝ := 180

theorem elaine_rent_percentage :
  last_year_rent_percentage = 20 :=
by
  sorry

#check elaine_rent_percentage

end NUMINAMATH_CALUDE_elaine_rent_percentage_l639_63939


namespace NUMINAMATH_CALUDE_min_removals_for_no_products_l639_63967

theorem min_removals_for_no_products (n : ℕ) (hn : n = 1982) :
  ∃ (S : Finset ℕ),
    S.card = 43 ∧ 
    (∀ k ∈ Finset.range (n + 1) \ S, k = 1 ∨ k ≥ 45) ∧
    (∀ a b k, a ∈ Finset.range (n + 1) \ S → b ∈ Finset.range (n + 1) \ S → 
      k ∈ Finset.range (n + 1) \ S → a ≠ b → a * b ≠ k) ∧
    (∀ T : Finset ℕ, T.card < 43 → 
      ∃ a b k, a ∈ Finset.range (n + 1) \ T → b ∈ Finset.range (n + 1) \ T → 
        k ∈ Finset.range (n + 1) \ T → a ≠ b → a * b = k) :=
by sorry

end NUMINAMATH_CALUDE_min_removals_for_no_products_l639_63967


namespace NUMINAMATH_CALUDE_polynomial_factorization_l639_63926

theorem polynomial_factorization (z : ℂ) : z^6 - 64*z^2 = z^2 * (z^2 - 8) * (z^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l639_63926


namespace NUMINAMATH_CALUDE_divisor_sum_difference_bound_l639_63968

/-- Sum of counts of positive even divisors of numbers from 1 to n -/
def D1 (n : ℕ) : ℕ := sorry

/-- Sum of counts of positive odd divisors of numbers from 1 to n -/
def D2 (n : ℕ) : ℕ := sorry

/-- The difference between D2 and D1 is no greater than n -/
theorem divisor_sum_difference_bound (n : ℕ) : D2 n - D1 n ≤ n := by sorry

end NUMINAMATH_CALUDE_divisor_sum_difference_bound_l639_63968


namespace NUMINAMATH_CALUDE_limit_of_ratio_l639_63928

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def sum_of_terms (n : ℕ) : ℝ := n^2

theorem limit_of_ratio :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sum_of_terms n / (arithmetic_sequence n)^2 - 1/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_ratio_l639_63928


namespace NUMINAMATH_CALUDE_vector_parallel_implies_m_value_l639_63904

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem vector_parallel_implies_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2, 1 + m)
  let b : ℝ × ℝ := (3, m)
  parallel a b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_m_value_l639_63904


namespace NUMINAMATH_CALUDE_beshmi_investment_l639_63930

theorem beshmi_investment (savings : ℝ) : 
  (1 / 5 : ℝ) * savings + 0.42 * savings + (savings - (1 / 5 : ℝ) * savings - 0.42 * savings) = savings
    → 0.42 * savings = 10500
    → savings - (1 / 5 : ℝ) * savings - 0.42 * savings = 9500 :=
by
  sorry

end NUMINAMATH_CALUDE_beshmi_investment_l639_63930


namespace NUMINAMATH_CALUDE_shoe_probability_l639_63945

def total_pairs : ℕ := 20
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 5
def gray_pairs : ℕ := 3
def white_pairs : ℕ := 4

theorem shoe_probability :
  let total_shoes := total_pairs * 2
  let prob_black := (black_pairs * 2 / total_shoes) * (black_pairs / (total_shoes - 1))
  let prob_brown := (brown_pairs * 2 / total_shoes) * (brown_pairs / (total_shoes - 1))
  let prob_gray := (gray_pairs * 2 / total_shoes) * (gray_pairs / (total_shoes - 1))
  let prob_white := (white_pairs * 2 / total_shoes) * (white_pairs / (total_shoes - 1))
  prob_black + prob_brown + prob_gray + prob_white = 19 / 130 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_l639_63945


namespace NUMINAMATH_CALUDE_class_average_calculation_l639_63978

theorem class_average_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_average : ℚ)
  (group2_students : ℕ) (group2_average : ℚ) :
  total_students = 30 →
  group1_students = 24 →
  group2_students = 6 →
  group1_average = 85 / 100 →
  group2_average = 92 / 100 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 864 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_average_calculation_l639_63978


namespace NUMINAMATH_CALUDE_greatest_constant_inequality_l639_63917

theorem greatest_constant_inequality (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

#check greatest_constant_inequality

end NUMINAMATH_CALUDE_greatest_constant_inequality_l639_63917


namespace NUMINAMATH_CALUDE_specific_window_side_length_l639_63985

/-- Represents a square window with glass panes -/
structure SquareWindow where
  /-- Number of panes in each row/column -/
  panes_per_side : ℕ
  /-- Width of a single pane -/
  pane_width : ℝ
  /-- Width of borders between panes and around the window -/
  border_width : ℝ

/-- Calculates the side length of the square window -/
def window_side_length (w : SquareWindow) : ℝ :=
  w.panes_per_side * w.pane_width + (w.panes_per_side + 1) * w.border_width

/-- Theorem stating the side length of the specific window described in the problem -/
theorem specific_window_side_length :
  ∃ w : SquareWindow,
    w.panes_per_side = 3 ∧
    w.pane_width * 3 = w.pane_width * w.panes_per_side ∧
    w.border_width = 3 ∧
    window_side_length w = 42 := by
  sorry

end NUMINAMATH_CALUDE_specific_window_side_length_l639_63985


namespace NUMINAMATH_CALUDE_maze_exit_probabilities_l639_63933

/-- Represents the three passages in the maze -/
inductive Passage
| one
| two
| three

/-- Time taken to exit each passage -/
def exit_time (p : Passage) : ℕ :=
  match p with
  | Passage.one => 1
  | Passage.two => 2
  | Passage.three => 3

/-- The probability of selecting a passage when n passages are available -/
def select_prob (n : ℕ) : ℚ :=
  1 / n

theorem maze_exit_probabilities :
  let p_one_hour := select_prob 3
  let p_more_than_three_hours := 
    select_prob 3 * select_prob 2 + 
    select_prob 3 * select_prob 2 + 
    select_prob 3 * select_prob 2
  (p_one_hour = 1/3) ∧ 
  (p_more_than_three_hours = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_maze_exit_probabilities_l639_63933


namespace NUMINAMATH_CALUDE_cell_count_after_twelve_days_l639_63948

/-- Represents the cell growth and death process over 12 days -/
def cell_growth (initial_cells : ℕ) (split_interval : ℕ) (total_days : ℕ) (death_day : ℕ) (cells_died : ℕ) : ℕ :=
  let cycles := total_days / split_interval
  let final_count := initial_cells * 2^cycles
  if death_day ≤ total_days then final_count - cells_died else final_count

/-- Theorem stating the number of cells after 12 days -/
theorem cell_count_after_twelve_days :
  cell_growth 5 3 12 9 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_cell_count_after_twelve_days_l639_63948


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l639_63934

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l639_63934


namespace NUMINAMATH_CALUDE_f_2017_negative_two_equals_three_fifths_l639_63940

def f (x : ℚ) : ℚ := (x - 1) / (3 * x + 1)

def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem f_2017_negative_two_equals_three_fifths :
  iterate_f 2017 (-2 : ℚ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_negative_two_equals_three_fifths_l639_63940


namespace NUMINAMATH_CALUDE_total_students_l639_63923

/-- The number of students who went to the movie -/
def M : ℕ := 10

/-- The number of students who went to the picnic -/
def P : ℕ := 20

/-- The number of students who played games -/
def G : ℕ := 5

/-- The number of students who went to both the movie and the picnic -/
def MP : ℕ := 4

/-- The number of students who went to both the movie and games -/
def MG : ℕ := 2

/-- The number of students who went to both the picnic and games -/
def PG : ℕ := 0

/-- The number of students who participated in all three activities -/
def MPG : ℕ := 2

/-- The total number of students -/
def T : ℕ := M + P + G - MP - MG - PG + MPG

theorem total_students : T = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l639_63923


namespace NUMINAMATH_CALUDE_expand_expression_l639_63941

theorem expand_expression (x : ℝ) : (2*x - 3) * (2*x + 3) * (4*x^2 + 9) = 4*x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l639_63941


namespace NUMINAMATH_CALUDE_not_always_perfect_square_exists_l639_63964

/-- Given an n-digit number x, prove that there doesn't always exist a non-negative integer y ≤ 9
    and an integer z such that 10^(n+1) * z + 10x + y is a perfect square. -/
theorem not_always_perfect_square_exists (n : ℕ) : 
  ∃ x : ℕ, (10^n ≤ x ∧ x < 10^(n+1)) →
    ¬∃ (y z : ℤ), 0 ≤ y ∧ y ≤ 9 ∧ ∃ (k : ℤ), 10^(n+1) * z + 10 * x + y = k^2 :=
by sorry

end NUMINAMATH_CALUDE_not_always_perfect_square_exists_l639_63964


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l639_63969

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℤ) - x * (⌊x⌋ : ℤ) = 7 ∧ 
  ∀ (y : ℝ), y > 0 → (⌊y^2⌋ : ℤ) - y * (⌊y⌋ : ℤ) = 7 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l639_63969


namespace NUMINAMATH_CALUDE_prime_divisor_count_l639_63902

theorem prime_divisor_count (p : ℕ) (hp : Prime p) 
  (h : ∃ k : ℤ, (28^p - 1 : ℤ) = k * (2*p^2 + 2*p + 1)) : 
  Prime (2*p^2 + 2*p + 1) := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_count_l639_63902


namespace NUMINAMATH_CALUDE_chef_cooked_seven_potatoes_l639_63918

/-- Represents the cooking scenario of a chef with potatoes -/
structure PotatoCookingScenario where
  total_potatoes : ℕ
  cooking_time_per_potato : ℕ
  remaining_cooking_time : ℕ

/-- Calculates the number of potatoes already cooked -/
def potatoes_already_cooked (scenario : PotatoCookingScenario) : ℕ :=
  scenario.total_potatoes - (scenario.remaining_cooking_time / scenario.cooking_time_per_potato)

/-- Theorem stating that the chef has already cooked 7 potatoes -/
theorem chef_cooked_seven_potatoes (scenario : PotatoCookingScenario)
  (h1 : scenario.total_potatoes = 16)
  (h2 : scenario.cooking_time_per_potato = 5)
  (h3 : scenario.remaining_cooking_time = 45) :
  potatoes_already_cooked scenario = 7 := by
  sorry

#eval potatoes_already_cooked { total_potatoes := 16, cooking_time_per_potato := 5, remaining_cooking_time := 45 }

end NUMINAMATH_CALUDE_chef_cooked_seven_potatoes_l639_63918


namespace NUMINAMATH_CALUDE_at_least_one_nonnegative_l639_63912

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  (ac + bd ≥ 0) ∨ (ae + bf ≥ 0) ∨ (ag + bh ≥ 0) ∨ 
  (ce + df ≥ 0) ∨ (cg + dh ≥ 0) ∨ (eg + fh ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_nonnegative_l639_63912


namespace NUMINAMATH_CALUDE_balloon_permutations_l639_63965

def balloon_letters : Nat := 7
def balloon_l_count : Nat := 2
def balloon_o_count : Nat := 2

theorem balloon_permutations :
  (balloon_letters.factorial) / (balloon_l_count.factorial * balloon_o_count.factorial) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l639_63965


namespace NUMINAMATH_CALUDE_max_height_is_three_l639_63954

/-- Represents a rectangular prism formed by unit cubes -/
structure RectangularPrism where
  base_area : ℕ
  height : ℕ

/-- The volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℕ :=
  prism.base_area * prism.height

/-- The set of all possible rectangular prisms with a given base area -/
def possible_prisms (base_area : ℕ) (total_cubes : ℕ) : Set RectangularPrism :=
  {prism | prism.base_area = base_area ∧ volume prism ≤ total_cubes}

/-- The theorem stating that the maximum height of a rectangular prism
    with base area 4 and 12 total cubes is 3 -/
theorem max_height_is_three :
  ∀ (prism : RectangularPrism),
    prism ∈ possible_prisms 4 12 →
    prism.height ≤ 3 ∧
    ∃ (max_prism : RectangularPrism),
      max_prism ∈ possible_prisms 4 12 ∧
      max_prism.height = 3 :=
sorry

end NUMINAMATH_CALUDE_max_height_is_three_l639_63954


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_holds_l639_63949

theorem min_value_of_function (x : ℝ) (h : x > 0) : 3 * x + 12 / x^2 ≥ 9 := by
  sorry

theorem equality_holds : ∃ x : ℝ, x > 0 ∧ 3 * x + 12 / x^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_holds_l639_63949


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l639_63922

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  ((x₁ + 1) * (x₁ - 1) = 2 * x₁ + 3) ∧ ((x₂ + 1) * (x₂ - 1) = 2 * x₂ + 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l639_63922


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l639_63982

/-- Given two vectors a and b in R^3, if a is parallel to b, then m = -2 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 3 → ℝ := ![2*m+1, 3, m-1]
  let b : Fin 3 → ℝ := ![2, m, -m]
  (∃ (k : ℝ), a = k • b) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l639_63982


namespace NUMINAMATH_CALUDE_find_number_l639_63976

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 57 :=
  sorry

end NUMINAMATH_CALUDE_find_number_l639_63976


namespace NUMINAMATH_CALUDE_polly_tweets_theorem_l639_63987

/-- Polly's tweeting behavior -/
structure PollyTweets where
  happy_rate : ℕ
  hungry_rate : ℕ
  mirror_rate : ℕ
  happy_duration : ℕ
  hungry_duration : ℕ
  mirror_duration : ℕ

/-- Calculate the total number of tweets -/
def total_tweets (p : PollyTweets) : ℕ :=
  p.happy_rate * p.happy_duration +
  p.hungry_rate * p.hungry_duration +
  p.mirror_rate * p.mirror_duration

/-- Theorem: Polly's total tweets equal 1340 -/
theorem polly_tweets_theorem (p : PollyTweets) 
  (h1 : p.happy_rate = 18)
  (h2 : p.hungry_rate = 4)
  (h3 : p.mirror_rate = 45)
  (h4 : p.happy_duration = 20)
  (h5 : p.hungry_duration = 20)
  (h6 : p.mirror_duration = 20) :
  total_tweets p = 1340 := by
  sorry

end NUMINAMATH_CALUDE_polly_tweets_theorem_l639_63987


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l639_63908

theorem largest_integer_satisfying_inequality :
  ∀ n : ℕ, n^200 < 5^300 ↔ n ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l639_63908


namespace NUMINAMATH_CALUDE_power_multiplication_l639_63995

theorem power_multiplication (m : ℝ) : m^5 * m = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l639_63995


namespace NUMINAMATH_CALUDE_emilee_earnings_l639_63951

/-- Given the total earnings and individual earnings of Jermaine and Terrence, 
    calculate Emilee's earnings. -/
theorem emilee_earnings 
  (total : ℕ) 
  (terrence_earnings : ℕ) 
  (jermaine_terrence_diff : ℕ) 
  (h1 : total = 90) 
  (h2 : terrence_earnings = 30) 
  (h3 : jermaine_terrence_diff = 5) : 
  total - (terrence_earnings + (terrence_earnings + jermaine_terrence_diff)) = 25 :=
by
  sorry

#check emilee_earnings

end NUMINAMATH_CALUDE_emilee_earnings_l639_63951


namespace NUMINAMATH_CALUDE_equation_solution_l639_63938

theorem equation_solution : ∃ (x₁ x₂ : ℚ), x₁ = 7/4 ∧ x₂ = 1/4 ∧
  (16 * (x₁ - 1)^2 - 9 = 0) ∧ (16 * (x₂ - 1)^2 - 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l639_63938


namespace NUMINAMATH_CALUDE_line_passes_through_points_l639_63929

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The two given points -/
def p1 : Point := ⟨-1, 1⟩
def p2 : Point := ⟨3, 9⟩

/-- The line we want to prove passes through the given points -/
def line : Line := ⟨2, -1, 3⟩

/-- Theorem stating that the given line passes through both points -/
theorem line_passes_through_points : 
  p1.liesOn line ∧ p2.liesOn line := by sorry

end NUMINAMATH_CALUDE_line_passes_through_points_l639_63929


namespace NUMINAMATH_CALUDE_quadratic_counterexample_l639_63911

theorem quadratic_counterexample :
  ∃ m : ℝ, m < -2 ∧ ∀ x : ℝ, x^2 + m*x + 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_counterexample_l639_63911


namespace NUMINAMATH_CALUDE_reading_pattern_equation_l639_63944

/-- Represents the total number of words in the book "Mencius" -/
def total_words : ℕ := 34685

/-- Represents the number of days taken to read the book -/
def days : ℕ := 3

/-- Represents the relationship between words read on consecutive days -/
def daily_increase_factor : ℕ := 2

/-- Theorem stating the correct equation for the reading pattern -/
theorem reading_pattern_equation (x : ℕ) :
  x + daily_increase_factor * x + daily_increase_factor^2 * x = total_words →
  x + 2*x + 4*x = total_words :=
by sorry

end NUMINAMATH_CALUDE_reading_pattern_equation_l639_63944


namespace NUMINAMATH_CALUDE_problem_solution_l639_63914

theorem problem_solution (m n : ℝ) : 
  (Real.sqrt (1 - m))^2 + |n + 2| = 0 → m - n = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l639_63914


namespace NUMINAMATH_CALUDE_expense_calculation_correct_l639_63936

/-- Calculates the total out-of-pocket expense for James' purchases and transactions --/
def total_expense (initial_purchase : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
  (tv_cost : ℝ) (bike_cost : ℝ) (usd_to_eur_initial : ℝ) (usd_to_eur_refund : ℝ)
  (usd_to_gbp : ℝ) (other_bike_markup : ℝ) (other_bike_sale_rate : ℝ)
  (toaster_cost_eur : ℝ) (microwave_cost_eur : ℝ)
  (subscription_cost_gbp : ℝ) (subscription_discount : ℝ) (subscription_months : ℕ) : ℝ :=
  sorry

/-- The total out-of-pocket expense matches the calculated value --/
theorem expense_calculation_correct :
  total_expense 5000 0.1 0.05 1000 700 0.85 0.87 0.77 0.2 0.85 100 150 80 0.3 12 = 2291.63 :=
  sorry

end NUMINAMATH_CALUDE_expense_calculation_correct_l639_63936


namespace NUMINAMATH_CALUDE_input_statement_is_INPUT_l639_63986

-- Define the possible statement types
inductive Statement
  | PRINT
  | INPUT
  | IF
  | WHILE

-- Define the function of each statement
def statementFunction (s : Statement) : String :=
  match s with
  | Statement.PRINT => "output"
  | Statement.INPUT => "input"
  | Statement.IF => "conditional execution"
  | Statement.WHILE => "looping"

-- Theorem to prove
theorem input_statement_is_INPUT :
  ∃ s : Statement, statementFunction s = "input" ∧ s = Statement.INPUT :=
  sorry

end NUMINAMATH_CALUDE_input_statement_is_INPUT_l639_63986


namespace NUMINAMATH_CALUDE_complex_magnitude_l639_63983

theorem complex_magnitude (z : ℂ) (h : Complex.I * z + 2 = z - 2 * Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l639_63983


namespace NUMINAMATH_CALUDE_baking_distribution_problem_l639_63920

/-- Calculates the number of leftover items when distributing a total number of items into containers of a specific capacity -/
def leftovers (total : ℕ) (capacity : ℕ) : ℕ :=
  total % capacity

/-- Represents the baking and distribution problem -/
theorem baking_distribution_problem 
  (gingerbread_batches : ℕ) (gingerbread_per_batch : ℕ) (gingerbread_per_jar : ℕ)
  (sugar_batches : ℕ) (sugar_per_batch : ℕ) (sugar_per_box : ℕ)
  (tart_batches : ℕ) (tarts_per_batch : ℕ) (tarts_per_box : ℕ)
  (h_gingerbread : gingerbread_batches = 3 ∧ gingerbread_per_batch = 47 ∧ gingerbread_per_jar = 6)
  (h_sugar : sugar_batches = 2 ∧ sugar_per_batch = 78 ∧ sugar_per_box = 9)
  (h_tart : tart_batches = 4 ∧ tarts_per_batch = 36 ∧ tarts_per_box = 4) :
  leftovers (gingerbread_batches * gingerbread_per_batch) gingerbread_per_jar = 3 ∧
  leftovers (sugar_batches * sugar_per_batch) sugar_per_box = 3 ∧
  leftovers (tart_batches * tarts_per_batch) tarts_per_box = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_baking_distribution_problem_l639_63920


namespace NUMINAMATH_CALUDE_hackathon_ends_at_noon_l639_63962

-- Define the start time of the hackathon
def hackathon_start : Nat := 12 * 60  -- noon in minutes since midnight

-- Define the duration of the hackathon
def hackathon_duration : Nat := 1440  -- duration in minutes

-- Define a function to calculate the end time of the hackathon
def hackathon_end (start : Nat) (duration : Nat) : Nat :=
  (start + duration) % (24 * 60)

-- Theorem to prove
theorem hackathon_ends_at_noon :
  hackathon_end hackathon_start hackathon_duration = hackathon_start :=
by sorry

end NUMINAMATH_CALUDE_hackathon_ends_at_noon_l639_63962


namespace NUMINAMATH_CALUDE_similar_triangle_coordinates_l639_63970

/-- Given two points A and B in a Cartesian coordinate system, with O as the origin and center of
    similarity, and triangle A'B'O similar to triangle ABO with a similarity ratio of 1:2,
    prove that the coordinates of B' are (-3, -2). -/
theorem similar_triangle_coordinates (A B : ℝ × ℝ) (h_A : A = (-4, 2)) (h_B : B = (-6, -4)) :
  let O : ℝ × ℝ := (0, 0)
  let similarity_ratio : ℝ := 1 / 2
  let B' : ℝ × ℝ := (similarity_ratio * B.1, similarity_ratio * B.2)
  B' = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_coordinates_l639_63970


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l639_63966

theorem consecutive_integers_problem (x y z : ℤ) : 
  x = y + 1 → 
  y = z + 1 → 
  x > y → 
  y > z → 
  2*x + 3*y + 3*z = 5*y + 8 → 
  z = 2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l639_63966


namespace NUMINAMATH_CALUDE_red_flower_area_is_54_total_area_red_yellow_equal_red_yellow_half_total_l639_63980

/-- Represents a rectangular plot with flowers and grass -/
structure FlowerPlot where
  length : ℝ
  width : ℝ
  red_flower_area : ℝ
  yellow_flower_area : ℝ
  grass_area : ℝ

/-- The properties of the flower plot as described in the problem -/
def school_plot : FlowerPlot where
  length := 18
  width := 12
  red_flower_area := 54
  yellow_flower_area := 54
  grass_area := 108

/-- Theorem stating that the area of red flowers in the school plot is 54 square meters -/
theorem red_flower_area_is_54 (plot : FlowerPlot) (h1 : plot = school_plot) :
  plot.red_flower_area = 54 := by
  sorry

/-- Theorem stating that the total area of the plot is length * width -/
theorem total_area (plot : FlowerPlot) : 
  plot.length * plot.width = plot.red_flower_area + plot.yellow_flower_area + plot.grass_area := by
  sorry

/-- Theorem stating that red and yellow flower areas are equal -/
theorem red_yellow_equal (plot : FlowerPlot) :
  plot.red_flower_area = plot.yellow_flower_area := by
  sorry

/-- Theorem stating that red and yellow flowers together occupy half the total area -/
theorem red_yellow_half_total (plot : FlowerPlot) :
  plot.red_flower_area + plot.yellow_flower_area = (plot.length * plot.width) / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_flower_area_is_54_total_area_red_yellow_equal_red_yellow_half_total_l639_63980


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l639_63932

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k) ↔ 
  (∃ m : ℤ, p = 3*m + 2) ∧ (∃ n : ℤ, q = 3*n) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l639_63932


namespace NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l639_63994

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

def satisfies_condition (n : ℕ) : Prop :=
  n < 500 ∧ n = 7 * sum_of_digits n ∧ is_prime (sum_of_digits n)

theorem exactly_two_numbers_satisfy :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l639_63994


namespace NUMINAMATH_CALUDE_sequence_relation_l639_63913

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : (y n)^2 = 3 * (x n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l639_63913


namespace NUMINAMATH_CALUDE_probability_calculation_l639_63984

theorem probability_calculation (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) 
  (remaining : ℕ) (h1 : total_students = 2006) (h2 : eliminated = 6) 
  (h3 : selected = 50) (h4 : remaining = total_students - eliminated) :
  (eliminated : ℚ) / (total_students : ℚ) = 3 / 1003 ∧ 
  (selected : ℚ) / (remaining : ℚ) = 25 / 1003 := by
  sorry

#check probability_calculation

end NUMINAMATH_CALUDE_probability_calculation_l639_63984


namespace NUMINAMATH_CALUDE_cyclist_return_speed_l639_63993

/-- Calculates the average speed for the return trip of a cyclist -/
theorem cyclist_return_speed (total_distance : ℝ) (first_half_distance : ℝ) (first_speed : ℝ) (second_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 32)
  (h2 : first_half_distance = 16)
  (h3 : first_speed = 8)
  (h4 : second_speed = 10)
  (h5 : total_time = 6.8)
  : (total_distance / (total_time - (first_half_distance / first_speed + (total_distance - first_half_distance) / second_speed))) = 10 := by
  sorry

#check cyclist_return_speed

end NUMINAMATH_CALUDE_cyclist_return_speed_l639_63993


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l639_63979

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center := reflect_about_neg_x original_center
  reflected_center = (3, -8) := by
sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l639_63979


namespace NUMINAMATH_CALUDE_distribution_schemes_l639_63915

def math_teachers : ℕ := 3
def chinese_teachers : ℕ := 6
def schools : ℕ := 3
def math_teachers_per_school : ℕ := 1
def chinese_teachers_per_school : ℕ := 2

theorem distribution_schemes :
  (math_teachers.factorial) *
  (chinese_teachers.choose chinese_teachers_per_school) *
  ((chinese_teachers - chinese_teachers_per_school).choose chinese_teachers_per_school) = 540 :=
sorry

end NUMINAMATH_CALUDE_distribution_schemes_l639_63915


namespace NUMINAMATH_CALUDE_happy_cattle_ranch_population_l639_63952

/-- The number of cows after n years, given an initial population and growth rate -/
def cowPopulation (initialPopulation : ℕ) (growthRate : ℚ) (years : ℕ) : ℚ :=
  initialPopulation * (1 + growthRate) ^ years

/-- Theorem: The cow population on Happy Cattle Ranch after 2 years -/
theorem happy_cattle_ranch_population :
  cowPopulation 200 (1/2) 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_happy_cattle_ranch_population_l639_63952


namespace NUMINAMATH_CALUDE_points_on_circle_l639_63942

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a motion (isometry) of the plane -/
structure Motion where
  transform : Point → Point

/-- A system of points with the given property -/
structure PointSystem where
  points : List Point
  motion_property : ∀ (p q : Point), p ∈ points → q ∈ points → 
    ∃ (m : Motion), m.transform p = q ∧ (∀ x ∈ points, m.transform x ∈ points)

/-- Definition of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The main theorem -/
theorem points_on_circle (sys : PointSystem) : 
  ∃ (c : Circle), ∀ p ∈ sys.points, 
    (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_points_on_circle_l639_63942


namespace NUMINAMATH_CALUDE_interest_rate_difference_l639_63903

/-- Proves that the difference in interest rates is 3% given the problem conditions -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_difference : ℝ)
  (h_principal : principal = 5000)
  (h_time : time = 2)
  (h_interest_diff : interest_difference = 300)
  : ∃ (r dr : ℝ),
    principal * (r + dr) / 100 * time - principal * r / 100 * time = interest_difference ∧
    dr = 3 := by
  sorry

#check interest_rate_difference

end NUMINAMATH_CALUDE_interest_rate_difference_l639_63903
