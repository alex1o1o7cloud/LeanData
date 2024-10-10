import Mathlib

namespace sphere_radius_increase_l1045_104505

theorem sphere_radius_increase (r : ℝ) (h : r > 0) : 
  let A := 4 * Real.pi * r^2
  let r' := Real.sqrt (2.25 * r^2)
  let A' := 4 * Real.pi * r'^2
  A' = 2.25 * A → r' = 1.5 * r :=
by sorry

end sphere_radius_increase_l1045_104505


namespace sum_of_integers_l1045_104529

theorem sum_of_integers (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x = y + 3) (h4 : x^3 - y^3 = 63) :
  x + y = 5 := by sorry

end sum_of_integers_l1045_104529


namespace adjacent_sum_divisible_by_four_l1045_104564

/-- A board is a 22x22 grid of natural numbers -/
def Board : Type := Fin 22 → Fin 22 → ℕ

/-- A cell is a position on the board -/
def Cell : Type := Fin 22 × Fin 22

/-- Two cells are adjacent if they share a side or vertex -/
def adjacent (c1 c2 : Cell) : Prop :=
  let (x1, y1) := c1
  let (x2, y2) := c2
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val) ∨
  (x1.val + 1 = x2.val ∧ y1.val + 1 = y2.val) ∨
  (x2.val + 1 = x1.val ∧ y1.val + 1 = y2.val) ∨
  (x1.val + 1 = x2.val ∧ y2.val + 1 = y1.val) ∨
  (x2.val + 1 = x1.val ∧ y2.val + 1 = y1.val)

/-- A valid board contains numbers from 1 to 22² -/
def valid_board (b : Board) : Prop :=
  ∀ x y, 1 ≤ b x y ∧ b x y ≤ 22^2

theorem adjacent_sum_divisible_by_four (b : Board) (h : valid_board b) :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (b c1.1 c1.2 + b c2.1 c2.2) % 4 = 0 :=
sorry

end adjacent_sum_divisible_by_four_l1045_104564


namespace cloth_sale_cost_price_l1045_104544

/-- Given the conditions of a cloth sale, prove the cost price per metre -/
theorem cloth_sale_cost_price 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (discount_rate : ℚ) 
  (tax_rate : ℚ) 
  (h1 : total_metres = 300)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_metre = 5)
  (h4 : discount_rate = 1/10)
  (h5 : tax_rate = 1/20)
  : ℕ := by
  sorry

#check cloth_sale_cost_price

end cloth_sale_cost_price_l1045_104544


namespace equal_selection_probability_l1045_104582

/-- Represents a two-stage sampling process -/
structure TwoStageSampling where
  initial_count : ℕ
  excluded_count : ℕ
  selected_count : ℕ

/-- Calculates the probability of selection in a two-stage sampling process -/
def selection_probability (sampling : TwoStageSampling) : ℚ :=
  sampling.selected_count / (sampling.initial_count - sampling.excluded_count)

/-- Theorem stating that the selection probability is equal for all students and is 50/2000 -/
theorem equal_selection_probability (sampling : TwoStageSampling) 
  (h1 : sampling.initial_count = 2011)
  (h2 : sampling.excluded_count = 11)
  (h3 : sampling.selected_count = 50) :
  selection_probability sampling = 50 / 2000 := by
  sorry

#eval selection_probability ⟨2011, 11, 50⟩

end equal_selection_probability_l1045_104582


namespace problem_solution_l1045_104589

theorem problem_solution (x y : ℝ) (n : ℝ) : 
  x = 3 → y = 1 → n = x - y^(x-(y+1)) → n = 2 := by
  sorry

end problem_solution_l1045_104589


namespace units_digit_of_l_squared_plus_two_to_l_l1045_104559

def l : ℕ := 15^2 + 2^15

theorem units_digit_of_l_squared_plus_two_to_l (l : ℕ) (h : l = 15^2 + 2^15) : 
  (l^2 + 2^l) % 10 = 7 := by
  sorry

end units_digit_of_l_squared_plus_two_to_l_l1045_104559


namespace ice_cream_volume_l1045_104516

/-- The volume of ice cream in a right circular cone topped with a hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = 48 * π :=
by
  sorry

end ice_cream_volume_l1045_104516


namespace temperature_conversion_l1045_104539

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 95 → t = 35 := by
  sorry

end temperature_conversion_l1045_104539


namespace trout_division_l1045_104588

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) :
  total_trout = 18 →
  num_people = 2 →
  trout_per_person = total_trout / num_people →
  trout_per_person = 9 :=
by sorry

end trout_division_l1045_104588


namespace max_of_4_2_neg5_l1045_104552

def find_max (a b c : Int) : Int :=
  let max1 := max a b
  max max1 c

theorem max_of_4_2_neg5 :
  find_max 4 2 (-5) = 4 := by
  sorry

end max_of_4_2_neg5_l1045_104552


namespace right_triangle_with_hypotenuse_65_l1045_104554

theorem right_triangle_with_hypotenuse_65 :
  ∃! (a b : ℕ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 25 :=
by sorry

end right_triangle_with_hypotenuse_65_l1045_104554


namespace existence_of_equal_elements_l1045_104583

theorem existence_of_equal_elements
  (p q n : ℕ+)
  (h_sum : p + q < n)
  (x : Fin (n + 1) → ℤ)
  (h_boundary : x 0 = 0 ∧ x (Fin.last n) = 0)
  (h_diff : ∀ i : Fin n, x i.succ - x i = p ∨ x i.succ - x i = -q) :
  ∃ i j : Fin (n + 1), i < j ∧ (i, j) ≠ (0, Fin.last n) ∧ x i = x j :=
by sorry

end existence_of_equal_elements_l1045_104583


namespace angle_sum_proof_l1045_104590

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = 2 * Real.sqrt 5 / 5) (h4 : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * π / 4 := by
sorry

end angle_sum_proof_l1045_104590


namespace custom_mul_solution_l1045_104515

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that if a * 4 = 9 under the custom multiplication, then a = 12.5 -/
theorem custom_mul_solution :
  ∃ a : ℝ, custom_mul a 4 = 9 ∧ a = 12.5 := by
  sorry

end custom_mul_solution_l1045_104515


namespace Q_trajectory_equation_l1045_104595

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line on which point P moves -/
def line_P (p : Point) : Prop :=
  2 * p.x - p.y + 3 = 0

/-- The fixed point M -/
def M : Point :=
  ⟨-1, 2⟩

/-- Q is on the extension of PM and PM = MQ -/
def Q_position (p q : Point) : Prop :=
  q.x - M.x = M.x - p.x ∧ q.y - M.y = M.y - p.y

/-- The trajectory of point Q -/
def Q_trajectory (q : Point) : Prop :=
  2 * q.x - q.y + 5 = 0

/-- Theorem: The trajectory of Q satisfies the equation 2x - y + 5 = 0 -/
theorem Q_trajectory_equation :
  ∀ p q : Point, line_P p → Q_position p q → Q_trajectory q :=
by sorry

end Q_trajectory_equation_l1045_104595


namespace equal_sides_implies_rhombus_l1045_104593

-- Define a quadrilateral
structure Quadrilateral :=
  (sides : Fin 4 → ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Theorem statement
theorem equal_sides_implies_rhombus (q : Quadrilateral) :
  (∀ i j : Fin 4, q.sides i = q.sides j) → is_rhombus q :=
by
  sorry


end equal_sides_implies_rhombus_l1045_104593


namespace oliver_ate_seventeen_fruits_l1045_104504

/-- The number of fruits Oliver ate -/
def fruits_eaten (initial_cherries initial_strawberries initial_blueberries
                  final_cherries final_strawberries final_blueberries : ℕ) : ℕ :=
  (initial_cherries - final_cherries) +
  (initial_strawberries - final_strawberries) +
  (initial_blueberries - final_blueberries)

/-- Theorem stating that Oliver ate 17 fruits in total -/
theorem oliver_ate_seventeen_fruits :
  fruits_eaten 16 10 20 6 8 15 = 17 := by
  sorry

end oliver_ate_seventeen_fruits_l1045_104504


namespace cyrus_remaining_pages_l1045_104587

/-- Calculates the remaining pages Cyrus needs to write -/
def remaining_pages (total_pages first_day second_day third_day fourth_day : ℕ) : ℕ :=
  total_pages - (first_day + second_day + third_day + fourth_day)

/-- Theorem stating that Cyrus needs to write 315 more pages -/
theorem cyrus_remaining_pages :
  remaining_pages 500 25 (2 * 25) (2 * (2 * 25)) 10 = 315 := by
  sorry

end cyrus_remaining_pages_l1045_104587


namespace parallel_vectors_l1045_104514

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

theorem parallel_vectors (k : ℝ) :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (2, 3)
  let c : ℝ × ℝ := (-2, k)
  IsParallel (a.1 + b.1, a.2 + b.2) c → k = -8 := by
  sorry

end parallel_vectors_l1045_104514


namespace representatives_count_l1045_104521

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys -/
def numBoys : ℕ := 4

/-- The number of girls -/
def numGirls : ℕ := 4

/-- The total number of representatives to be selected -/
def totalReps : ℕ := 3

/-- The minimum number of girls to be selected -/
def minGirls : ℕ := 2

theorem representatives_count :
  (choose numGirls 2 * choose numBoys 1) + (choose numGirls 3) = 28 := by sorry

end representatives_count_l1045_104521


namespace viewership_difference_l1045_104576

/-- The number of viewers for each game this week -/
structure ViewersThisWeek where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The total number of viewers last week -/
def viewersLastWeek : ℕ := 350

/-- The conditions for this week's viewership -/
def viewershipConditions (v : ViewersThisWeek) : Prop :=
  v.second = 80 ∧
  v.first = v.second - 20 ∧
  v.third = v.second + 15 ∧
  v.fourth = v.third + (v.third / 10)

/-- The theorem to prove -/
theorem viewership_difference (v : ViewersThisWeek) 
  (h : viewershipConditions v) : 
  v.first + v.second + v.third + v.fourth = viewersLastWeek - 10 := by
  sorry

end viewership_difference_l1045_104576


namespace fans_with_all_items_l1045_104597

def arena_capacity : ℕ := 5000
def tshirt_interval : ℕ := 100
def cap_interval : ℕ := 40
def brochure_interval : ℕ := 60

theorem fans_with_all_items : 
  (arena_capacity / (Nat.lcm (Nat.lcm tshirt_interval cap_interval) brochure_interval) : ℕ) = 8 := by
  sorry

end fans_with_all_items_l1045_104597


namespace min_transportation_cost_l1045_104508

-- Define the problem parameters
def total_items : ℕ := 320
def water_excess : ℕ := 80
def type_a_water_capacity : ℕ := 40
def type_a_veg_capacity : ℕ := 10
def type_b_capacity : ℕ := 20
def total_trucks : ℕ := 8
def type_a_cost : ℕ := 400
def type_b_cost : ℕ := 360

-- Define the transportation cost function
def transportation_cost (num_type_a : ℕ) : ℕ :=
  type_a_cost * num_type_a + type_b_cost * (total_trucks - num_type_a)

-- Theorem statement
theorem min_transportation_cost :
  ∃ (num_water num_veg : ℕ),
    num_water + num_veg = total_items ∧
    num_water - num_veg = water_excess ∧
    (∀ (num_type_a : ℕ),
      2 ≤ num_type_a ∧ num_type_a ≤ 4 →
      type_a_water_capacity * num_type_a + type_b_capacity * (total_trucks - num_type_a) ≥ num_water ∧
      type_a_veg_capacity * num_type_a + type_b_capacity * (total_trucks - num_type_a) ≥ num_veg) ∧
    (∀ (num_type_a : ℕ),
      2 ≤ num_type_a ∧ num_type_a ≤ 4 →
      transportation_cost 2 ≤ transportation_cost num_type_a) ∧
    transportation_cost 2 = 2960 := by
  sorry

end min_transportation_cost_l1045_104508


namespace complex_below_real_axis_l1045_104524

theorem complex_below_real_axis (t : ℝ) : 
  let z : ℂ := (2 * t^2 + 5 * t - 3) + (t^2 + 2 * t + 2) * I
  Complex.im z < 0 := by
sorry

end complex_below_real_axis_l1045_104524


namespace seafood_price_seafood_price_proof_l1045_104532

/-- The regular price for two pounds of seafood given a 75% discount and $4 discounted price for one pound -/
theorem seafood_price : ℝ → ℝ → ℝ → Prop :=
  fun discount_percent discounted_price_per_pound regular_price_two_pounds =>
    discount_percent = 75 ∧
    discounted_price_per_pound = 4 →
    regular_price_two_pounds = 32

/-- Proof of the seafood price theorem -/
theorem seafood_price_proof :
  seafood_price 75 4 32 := by
  sorry

end seafood_price_seafood_price_proof_l1045_104532


namespace sin_75_cos_75_double_l1045_104509

theorem sin_75_cos_75_double : 2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2 := by
  sorry

end sin_75_cos_75_double_l1045_104509


namespace function_lower_bound_l1045_104596

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 3/4) * Real.exp x - (b * Real.exp x) / (Real.exp x + 1)

theorem function_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Ici (-2 : ℝ), f a 1 x ≥ -5/4) → a = 1 := by
  sorry

end function_lower_bound_l1045_104596


namespace triangle_shape_l1045_104531

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- positive angles
  (A + B + C = 180) →        -- sum of angles in a triangle
  (A = 30 ∨ B = 30 ∨ C = 30) →  -- one angle is 30°
  (a = 2*b ∨ b = 2*c ∨ c = 2*a) →  -- one side is twice another
  (¬(A < 90 ∧ B < 90 ∧ C < 90) ∧ (C = 90 ∨ C > 90 ∨ B > 90)) := by
sorry

end triangle_shape_l1045_104531


namespace inequality_solution_set_l1045_104572

theorem inequality_solution_set (x : ℝ) : 
  (1/3 : ℝ) + |x - 11/48| < 1/2 ↔ x ∈ Set.Ioo (1/16 : ℝ) (19/48 : ℝ) :=
sorry

end inequality_solution_set_l1045_104572


namespace isabellas_hair_growth_l1045_104501

theorem isabellas_hair_growth (initial_length growth : ℕ) (h1 : initial_length = 18) (h2 : growth = 6) :
  initial_length + growth = 24 := by
  sorry

end isabellas_hair_growth_l1045_104501


namespace tangent_line_equation_l1045_104527

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 + 8 * x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (14 * x - y - 8 = 0) :=
by sorry

end tangent_line_equation_l1045_104527


namespace half_full_one_minute_before_end_l1045_104599

/-- Represents the filling process of a box with marbles -/
def FillingProcess (total_time : ℕ) : Type :=
  ℕ → ℝ

/-- The quantity doubles every minute -/
def DoublesEveryMinute (process : FillingProcess n) : Prop :=
  ∀ t, t < n → process (t + 1) = 2 * process t

/-- The process is complete at the total time -/
def CompleteAtEnd (process : FillingProcess n) : Prop :=
  process n = 1

/-- The box is half full at a given time -/
def HalfFullAt (process : FillingProcess n) (t : ℕ) : Prop :=
  process t = 1/2

theorem half_full_one_minute_before_end 
  (process : FillingProcess 10) 
  (h1 : DoublesEveryMinute process) 
  (h2 : CompleteAtEnd process) :
  HalfFullAt process 9 :=
sorry

end half_full_one_minute_before_end_l1045_104599


namespace max_y_value_l1045_104565

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 := by
  sorry

end max_y_value_l1045_104565


namespace quadratic_coefficient_l1045_104533

theorem quadratic_coefficient (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 20 = (x + m)^2 + 8) → 
  b = 4 * Real.sqrt 3 := by
  sorry

end quadratic_coefficient_l1045_104533


namespace unique_integer_l1045_104517

theorem unique_integer (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) : 
  x = 6 := by
sorry

end unique_integer_l1045_104517


namespace function_property_l1045_104594

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_neg (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def half_x_on_unit (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 1/2 * x

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : periodic_neg f) 
  (h3 : half_x_on_unit f) : 
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ k : ℤ, x = 4 * k - 1} := by
  sorry

end function_property_l1045_104594


namespace fraction_multiplication_l1045_104575

theorem fraction_multiplication : (1/2 + 5/6 - 7/12) * (-36) = -27 := by
  sorry

end fraction_multiplication_l1045_104575


namespace equation_solutions_l1045_104522

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1) ∧
  (∀ x : ℝ, 2*x^2 - 3*x - 1 = 0 ↔ x = (3 + Real.sqrt 17) / 4 ∨ x = (3 - Real.sqrt 17) / 4) :=
by sorry

end equation_solutions_l1045_104522


namespace hex_to_binary_max_bits_l1045_104519

theorem hex_to_binary_max_bits :
  ∀ (A B C D : Nat),
  A < 16 → B < 16 → C < 16 → D < 16 →
  ∃ (n : Nat),
  n ≤ 16 ∧
  A * 16^3 + B * 16^2 + C * 16^1 + D < 2^n :=
by sorry

end hex_to_binary_max_bits_l1045_104519


namespace percentage_calculation_l1045_104537

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 63 = 232.42 := by sorry

end percentage_calculation_l1045_104537


namespace fish_tank_problem_l1045_104555

theorem fish_tank_problem (tank1_goldfish tank2 tank3 : ℕ) : 
  tank1_goldfish = 7 →
  tank3 = 10 →
  tank2 = 3 * tank3 →
  tank2 = 2 * (tank1_goldfish + (tank1_beta : ℕ)) →
  tank1_beta = 8 := by
  sorry

end fish_tank_problem_l1045_104555


namespace quadratic_root_sum_l1045_104571

theorem quadratic_root_sum (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 2*m = 0 := by
  sorry

end quadratic_root_sum_l1045_104571


namespace no_valid_A_exists_l1045_104506

theorem no_valid_A_exists : ¬∃ (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧
  ∃ (x : ℕ), x^2 - (2*A)*x + (A+1)*0 = 0 := by
  sorry

end no_valid_A_exists_l1045_104506


namespace kaprekar_convergence_l1045_104586

/-- Reverses a four-digit number -/
def reverseNumber (n : Nat) : Nat :=
  sorry

/-- Rearranges digits of a four-digit number from largest to smallest -/
def rearrangeDigits (n : Nat) : Nat :=
  sorry

/-- Applies the Kaprekar transformation to a four-digit number -/
def kaprekarTransform (n : Nat) : Nat :=
  let m := rearrangeDigits n
  let r := reverseNumber m
  m - r

/-- Applies the Kaprekar transformation k times -/
def kaprekarTransformK (n : Nat) (k : Nat) : Nat :=
  sorry

theorem kaprekar_convergence (n : Nat) (h : n = 5298 ∨ n = 4852) :
  ∃ (k : Nat) (t : Nat), k = 7 ∧ t = 6174 ∧
    kaprekarTransformK n k = t ∧
    kaprekarTransform t = t :=
  sorry

end kaprekar_convergence_l1045_104586


namespace subset_condition_empty_intersection_l1045_104567

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for part 1
theorem subset_condition (m : ℝ) : A ⊆ B m ↔ m ≤ -2 := by sorry

-- Theorem for part 2
theorem empty_intersection (m : ℝ) : A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end subset_condition_empty_intersection_l1045_104567


namespace probability_odd_even_function_selection_l1045_104580

theorem probability_odd_even_function_selection :
  let total_functions : ℕ := 7
  let odd_functions : ℕ := 3
  let even_functions : ℕ := 3
  let neither_odd_nor_even : ℕ := 1
  let total_selections : ℕ := total_functions.choose 2
  let favorable_selections : ℕ := odd_functions * even_functions
  favorable_selections / total_selections = 3 / 7 := by
sorry

end probability_odd_even_function_selection_l1045_104580


namespace problem_solution_l1045_104566

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end problem_solution_l1045_104566


namespace arithmetic_sequence_a1_value_l1045_104551

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 1 + a 5 + a 7 + a 9 + a 13 = 100) ∧
  (a 6 - a 2 = 12)

/-- The theorem stating that a_1 = 2 for the given arithmetic sequence -/
theorem arithmetic_sequence_a1_value (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 1 = 2 := by
  sorry

end arithmetic_sequence_a1_value_l1045_104551


namespace bob_same_color_probability_l1045_104556

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of marbles each person takes -/
def marbles_taken : ℕ := 3

/-- Calculates the probability of Bob getting 3 marbles of the same color -/
def prob_bob_same_color : ℚ :=
  let total_marbles := marbles_per_color * num_colors
  let total_outcomes := (total_marbles.choose marbles_taken) * 
                        ((total_marbles - marbles_taken).choose marbles_taken) * 
                        ((total_marbles - 2*marbles_taken).choose marbles_taken)
  let favorable_outcomes := num_colors * ((total_marbles - marbles_taken).choose marbles_taken)
  favorable_outcomes / total_outcomes

theorem bob_same_color_probability :
  prob_bob_same_color = 1 / 28 := by sorry

end bob_same_color_probability_l1045_104556


namespace salary_increase_after_reduction_l1045_104579

theorem salary_increase_after_reduction (original_salary : ℝ) (h : original_salary > 0) :
  let reduced_salary := original_salary * (1 - 0.35)
  let increase_factor := (1 / 0.65) - 1
  reduced_salary * (1 + increase_factor) = original_salary := by
  sorry

#eval (1 / 0.65 - 1) * 100 -- To show the approximate percentage increase

end salary_increase_after_reduction_l1045_104579


namespace repeating_decimal_sum_l1045_104562

theorem repeating_decimal_sum (a b : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9) →
  (3 : ℚ) / 13 = 
    (a : ℚ) / 10 + (b : ℚ) / 100 + 
    (a : ℚ) / 1000 + (b : ℚ) / 10000 + 
    (a : ℚ) / 100000 + (b : ℚ) / 1000000 + 
    (a : ℚ) / 10000000 + (b : ℚ) / 100000000 + 
    (a : ℚ) / 1000000000 + (b : ℚ) / 10000000000 →
  a + b = 5 := by
sorry

end repeating_decimal_sum_l1045_104562


namespace complement_of_M_l1045_104500

def M : Set ℝ := {x | x + 3 > 0}

theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = {x : ℝ | x ≤ -3} := by sorry

end complement_of_M_l1045_104500


namespace negation_of_inequality_implication_is_true_l1045_104598

theorem negation_of_inequality_implication_is_true :
  ∀ (a b c : ℝ), (a ≤ b → a * c^2 ≤ b * c^2) := by
  sorry

end negation_of_inequality_implication_is_true_l1045_104598


namespace marie_messages_per_day_l1045_104545

/-- Represents the problem of calculating the number of messages read per day -/
def messages_read_per_day (initial_unread : ℕ) (new_messages_per_day : ℕ) (days_to_clear : ℕ) : ℕ :=
  (initial_unread + new_messages_per_day * days_to_clear) / days_to_clear

/-- Theorem stating that Marie reads 20 messages per day -/
theorem marie_messages_per_day :
  messages_read_per_day 98 6 7 = 20 := by
  sorry

#eval messages_read_per_day 98 6 7

end marie_messages_per_day_l1045_104545


namespace circles_intersection_m_range_l1045_104536

/-- Circle C₁ with equation x² + y² - 2mx + m² - 4 = 0 -/
def C₁ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 + m^2 - 4 = 0}

/-- Circle C₂ with equation x² + y² + 2x - 4my + 4m² - 8 = 0 -/
def C₂ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*m*p.2 + 4*m^2 - 8 = 0}

/-- The theorem stating that if C₁ and C₂ intersect, then m is in the specified range -/
theorem circles_intersection_m_range (m : ℝ) :
  (C₁ m ∩ C₂ m).Nonempty →
  m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := by
  sorry

end circles_intersection_m_range_l1045_104536


namespace tangent_equality_mod_180_l1045_104502

theorem tangent_equality_mod_180 (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (2530 * π / 180) → m = 10 := by
  sorry

end tangent_equality_mod_180_l1045_104502


namespace watermelon_weight_l1045_104574

theorem watermelon_weight (total_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : total_weight = 63)
  (h2 : half_removed_weight = 34) :
  let watermelon_weight := total_weight - half_removed_weight * 2
  watermelon_weight = 58 := by
sorry

end watermelon_weight_l1045_104574


namespace smallest_square_addition_l1045_104546

theorem smallest_square_addition (n : ℕ) (h : n = 2019) : 
  ∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m^2 ∧ 
  ∀ k : ℕ, k < 1 → ¬∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + k = m^2 :=
by sorry

end smallest_square_addition_l1045_104546


namespace negation_of_conjunction_l1045_104525

theorem negation_of_conjunction (p q : Prop) : ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by sorry

end negation_of_conjunction_l1045_104525


namespace polynomial_expansion_problem_l1045_104568

theorem polynomial_expansion_problem (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  (45 * p^8 * q^2 = 120 * p^7 * q^3) → 
  (p + q = 3/4) → 
  p = 6/11 := by
  sorry

end polynomial_expansion_problem_l1045_104568


namespace trig_identity_l1045_104535

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin (x - y) - Real.cos (x + y) * Real.cos (x - y) = -Real.cos (2 * x) := by
  sorry

end trig_identity_l1045_104535


namespace negation_of_universal_proposition_l1045_104585

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 1) := by
  sorry

end negation_of_universal_proposition_l1045_104585


namespace nonnegative_solutions_count_l1045_104592

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by
  sorry

end nonnegative_solutions_count_l1045_104592


namespace c_constant_when_n_doubled_l1045_104510

/-- Given positive constants e, R, and r, and a positive variable n,
    the function C(n) remains constant when n is doubled. -/
theorem c_constant_when_n_doubled
  (e R r : ℝ) (n : ℝ) 
  (he : e > 0) (hR : R > 0) (hr : r > 0) (hn : n > 0) :
  let C : ℝ → ℝ := fun n => (e^2 * n) / (R + n * r^2)
  C n = C (2 * n) := by
  sorry

end c_constant_when_n_doubled_l1045_104510


namespace division_of_decimals_l1045_104573

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end division_of_decimals_l1045_104573


namespace book_sale_gain_percentage_l1045_104584

/-- Proves that given a book with a cost price CP, if selling it for 0.9 * CP
    results in Rs. 720, and selling it for Rs. 880 results in a gain,
    then the percentage of gain is 10%. -/
theorem book_sale_gain_percentage
  (CP : ℝ)  -- Cost price of the book
  (h1 : 0.9 * CP = 720)  -- Selling at 10% loss gives Rs. 720
  (h2 : 880 > CP)  -- Selling at Rs. 880 results in a gain
  : (880 - CP) / CP * 100 = 10 := by
  sorry

end book_sale_gain_percentage_l1045_104584


namespace simplify_sqrt_180_l1045_104557

theorem simplify_sqrt_180 : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_180_l1045_104557


namespace complex_expression_evaluation_l1045_104528

theorem complex_expression_evaluation : 
  let z₁ : ℂ := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z₂ : ℂ := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z₃ : ℂ := 1 / (8 * Complex.I^3)
  z₁ + z₂ + z₃ = -1.6 + 0.125 * Complex.I := by
sorry

end complex_expression_evaluation_l1045_104528


namespace other_x_intercept_of_quadratic_l1045_104578

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (3, 7) and one x-intercept at (-2, 0),
    the x-coordinate of the other x-intercept is 8. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 7 + a * (x - 3)^2) →  -- Vertex form of quadratic with vertex (3, 7)
  (a * (-2)^2 + b * (-2) + c = 0) →                 -- (-2, 0) is an x-intercept
  ∃ x, x ≠ -2 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=  -- Other x-intercept exists and equals 8
by sorry

end other_x_intercept_of_quadratic_l1045_104578


namespace smallest_n_for_candy_purchase_l1045_104560

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 24 * m = Nat.lcm (Nat.lcm 18 16) 20 → n ≤ m) ∧
  24 * n = Nat.lcm (Nat.lcm 18 16) 20 ∧ n = 30 := by
  sorry

end smallest_n_for_candy_purchase_l1045_104560


namespace unique_function_property_l1045_104507

def FunctionProperty (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, n^2 + 4 * f n = f (f (f n))

theorem unique_function_property :
  ∃! f : ℤ → ℤ, FunctionProperty f ∧ ∀ n : ℤ, f n = n + 1 :=
sorry

end unique_function_property_l1045_104507


namespace jills_study_hours_l1045_104530

/-- Represents Jill's study schedule over three days -/
structure StudySchedule where
  day1 : ℝ  -- Hours studied on day 1
  day2 : ℝ  -- Hours studied on day 2
  day3 : ℝ  -- Hours studied on day 3

/-- The theorem representing Jill's study problem -/
theorem jills_study_hours (schedule : StudySchedule) :
  schedule.day2 = 2 * schedule.day1 ∧
  schedule.day3 = 2 * schedule.day1 - 1 ∧
  schedule.day1 + schedule.day2 + schedule.day3 = 9 →
  schedule.day1 = 2 :=
by sorry

end jills_study_hours_l1045_104530


namespace simplify_w_squared_series_l1045_104547

theorem simplify_w_squared_series (w : ℝ) : 
  3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 := by
  sorry

end simplify_w_squared_series_l1045_104547


namespace unique_solution_l1045_104526

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48) :
  a = 13 ∧ b = 11 ∧ c = 9 := by
  sorry

end unique_solution_l1045_104526


namespace largest_power_of_two_divisor_l1045_104520

theorem largest_power_of_two_divisor (n : ℕ) :
  (∃ (k : ℕ), 2^k ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋ ∧
    ∀ (m : ℕ), m > k → ¬(2^m ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋)) →
  (∃! (k : ℕ), k = n ∧ 2^k ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋ ∧
    ∀ (m : ℕ), m > k → ¬(2^m ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋)) :=
by sorry

#check largest_power_of_two_divisor

end largest_power_of_two_divisor_l1045_104520


namespace bowl_water_problem_l1045_104563

theorem bowl_water_problem (C : ℝ) (h1 : C > 0) :
  C / 2 + 4 = 0.7 * C → 0.7 * C = 14 := by
  sorry

end bowl_water_problem_l1045_104563


namespace a2_times_a6_eq_68_l1045_104561

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℤ := 4 * n^2 - 10 * n

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := S n - S (n-1)

/-- Theorem stating that a_2 * a_6 = 68 -/
theorem a2_times_a6_eq_68 : a 2 * a 6 = 68 := by
  sorry

end a2_times_a6_eq_68_l1045_104561


namespace rectangular_section_properties_l1045_104534

/-- A regular tetrahedron with unit edge length -/
structure UnitTetrahedron where
  -- Add necessary fields here

/-- A rectangular section of a tetrahedron -/
structure RectangularSection (T : UnitTetrahedron) where
  -- Add necessary fields here

/-- The perimeter of a rectangular section -/
def perimeter (T : UnitTetrahedron) (S : RectangularSection T) : ℝ :=
  sorry

/-- The area of a rectangular section -/
def area (T : UnitTetrahedron) (S : RectangularSection T) : ℝ :=
  sorry

theorem rectangular_section_properties (T : UnitTetrahedron) :
  (∀ S : RectangularSection T, perimeter T S = 2) ∧
  (∀ S : RectangularSection T, 0 ≤ area T S ∧ area T S ≤ 1/4) :=
sorry

end rectangular_section_properties_l1045_104534


namespace max_cut_trees_100x100_l1045_104550

/-- Represents a square grid of trees -/
structure TreeGrid where
  size : ℕ
  trees : Fin size → Fin size → Bool

/-- Checks if a tree can be cut without making any other cut tree visible -/
def canCutTree (grid : TreeGrid) (x y : Fin grid.size) : Bool := sorry

/-- Counts the maximum number of trees that can be cut in a grid -/
def maxCutTrees (grid : TreeGrid) : ℕ := sorry

/-- Theorem: In a 100x100 grid, the maximum number of trees that can be cut
    while ensuring no stump is visible from any other stump is 2500 -/
theorem max_cut_trees_100x100 :
  ∀ (grid : TreeGrid), grid.size = 100 → maxCutTrees grid = 2500 := by sorry

end max_cut_trees_100x100_l1045_104550


namespace coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1045_104541

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * 1^k * 1^(8-k)) = 256 ∧
  Nat.choose 8 3 = 56 :=
sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1045_104541


namespace range_of_a_l1045_104518

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Theorem statement
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (1 - a) > f (2 * a - 1)) :
  2 / 3 < a ∧ a < 1 := by
  sorry

end range_of_a_l1045_104518


namespace josh_final_wallet_amount_l1045_104549

-- Define the initial conditions
def initial_wallet_amount : ℝ := 300
def initial_investment : ℝ := 2000
def stock_price_increase : ℝ := 0.30

-- Define the function to calculate the final amount
def final_wallet_amount : ℝ :=
  initial_wallet_amount + initial_investment * (1 + stock_price_increase)

-- Theorem to prove
theorem josh_final_wallet_amount :
  final_wallet_amount = 2900 := by
  sorry

end josh_final_wallet_amount_l1045_104549


namespace smallest_x_y_sum_l1045_104540

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_x_y_sum (x y : ℕ) : 
  (x > 0 ∧ y > 0) →
  (is_square (450 * x)) →
  (is_cube (450 * y)) →
  (∀ x' : ℕ, x' > 0 → x' < x → ¬(is_square (450 * x'))) →
  (∀ y' : ℕ, y' > 0 → y' < y → ¬(is_cube (450 * y'))) →
  x = 2 ∧ y = 60 ∧ x + y = 62 :=
by sorry

end smallest_x_y_sum_l1045_104540


namespace solution_set_transformation_l1045_104569

/-- Given that the solution set of ax^2 + bx + c > 0 is (1, 2),
    prove that the solution set of cx^2 + bx + a > 0 is (1/2, 1) -/
theorem solution_set_transformation (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < 2) →
  (∀ x : ℝ, c*x^2 + b*x + a > 0 ↔ 1/2 < x ∧ x < 1) :=
sorry

end solution_set_transformation_l1045_104569


namespace hyperbola_equation_l1045_104511

/-- The standard equation of a hyperbola passing through specific points and sharing asymptotes with another hyperbola -/
theorem hyperbola_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), y^2 / a - x^2 / b = 1 ↔ 
    ((x = -3 ∧ y = 2 * Real.sqrt 7) ∨ 
     (x = -6 * Real.sqrt 2 ∧ y = -7) ∨ 
     (x = 2 ∧ y = 2 * Real.sqrt 3)) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), y^2 / a - x^2 / b = 1 ↔ x^2 / 4 - y^2 / 3 = k)) ∧
  a = 9 ∧ b = 12 :=
sorry

end hyperbola_equation_l1045_104511


namespace house_price_calculation_l1045_104503

theorem house_price_calculation (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.56 * P = 56000) : P = 100000 :=
by
  sorry

end house_price_calculation_l1045_104503


namespace num_winning_configurations_l1045_104581

/-- Represents a 4x4 tic-tac-toe board -/
def Board := Fin 4 → Fin 4 → Option Bool

/-- Represents a 3x3 section of the 4x4 board -/
def Section := Fin 3 → Fin 3 → Option Bool

/-- The number of 3x3 sections in a 4x4 board -/
def numSections : Nat := 4

/-- The number of ways to place X's in a winning 3x3 section for horizontal or vertical wins -/
def numXPlacementsRowCol : Nat := 18

/-- The number of ways to place X's in a winning 3x3 section for diagonal wins -/
def numXPlacementsDiag : Nat := 20

/-- The number of rows or columns in a 3x3 section -/
def numRowsOrCols : Nat := 6

/-- The number of diagonals in a 3x3 section -/
def numDiagonals : Nat := 2

/-- Calculates the total number of winning configurations in one 3x3 section -/
def winsIn3x3Section : Nat :=
  numRowsOrCols * numXPlacementsRowCol + numDiagonals * numXPlacementsDiag

/-- The main theorem: proves that the number of possible board configurations after Carl wins is 592 -/
theorem num_winning_configurations :
  (numSections * winsIn3x3Section) = 592 := by sorry

end num_winning_configurations_l1045_104581


namespace sum_a_b_equals_seven_l1045_104558

/-- Represents a four-digit number in the form 3a72 -/
def fourDigitNum (a : ℕ) : ℕ := 3000 + 100 * a + 72

/-- Checks if a number is divisible by 11 -/
def divisibleBy11 (n : ℕ) : Prop := n % 11 = 0

theorem sum_a_b_equals_seven :
  ∀ a b : ℕ,
  (a < 10) →
  (b < 10) →
  (fourDigitNum a + 895 = 4000 + 100 * b + 67) →
  divisibleBy11 (4000 + 100 * b + 67) →
  a + b = 7 := by
sorry

end sum_a_b_equals_seven_l1045_104558


namespace acid_mixing_problem_l1045_104577

/-- The largest integer concentration percentage achievable in the acid mixing problem -/
def largest_concentration : ℕ := 76

theorem acid_mixing_problem :
  ∀ r : ℕ,
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ (2.8 + 0.9 * x) / (4 + x) = r / 100) →
    r ≤ largest_concentration :=
by sorry

end acid_mixing_problem_l1045_104577


namespace bridge_weight_is_88_ounces_l1045_104570

/-- The weight of a toy bridge given the number of full soda cans, 
    the weight of soda in each can, the weight of an empty can, 
    and the number of additional empty cans. -/
def bridge_weight (full_cans : ℕ) (soda_weight : ℕ) (empty_can_weight : ℕ) (additional_empty_cans : ℕ) : ℕ :=
  (full_cans * (soda_weight + empty_can_weight)) + (additional_empty_cans * empty_can_weight)

/-- Theorem stating that the bridge must hold 88 ounces given the specified conditions. -/
theorem bridge_weight_is_88_ounces : 
  bridge_weight 6 12 2 2 = 88 := by
  sorry

end bridge_weight_is_88_ounces_l1045_104570


namespace geometric_mean_combined_sets_l1045_104553

theorem geometric_mean_combined_sets :
  ∀ (y₁ y₂ y₃ y₄ z₁ z₂ z₃ z₄ : ℝ),
    y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧
    z₁ > 0 ∧ z₂ > 0 ∧ z₃ > 0 ∧ z₄ > 0 →
    (y₁ * y₂ * y₃ * y₄) ^ (1/4 : ℝ) = 2048 →
    (z₁ * z₂ * z₃ * z₄) ^ (1/4 : ℝ) = 8 →
    (y₁ * y₂ * y₃ * y₄ * z₁ * z₂ * z₃ * z₄) ^ (1/8 : ℝ) = 128 :=
by
  sorry

end geometric_mean_combined_sets_l1045_104553


namespace teacher_score_calculation_l1045_104523

def teacher_total_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

theorem teacher_score_calculation :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  teacher_total_score written_score interview_score written_weight interview_weight = 72 := by
  sorry

end teacher_score_calculation_l1045_104523


namespace min_speed_against_current_l1045_104543

/-- The minimum speed against the current given the following conditions:
    - Man's speed with the current is 35 km/hr
    - Speed of the current varies between 5.6 km/hr and 8.4 km/hr
    - Wind resistance provides a decelerating force between 0.1 to 0.3 times his speed -/
theorem min_speed_against_current (speed_with_current : ℝ) 
  (current_speed_min current_speed_max : ℝ) 
  (wind_resistance_min wind_resistance_max : ℝ) :
  speed_with_current = 35 →
  current_speed_min = 5.6 →
  current_speed_max = 8.4 →
  wind_resistance_min = 0.1 →
  wind_resistance_max = 0.3 →
  ∃ (speed_against_current : ℝ), 
    speed_against_current ≥ 14.7 ∧ 
    (∀ (actual_current_speed actual_wind_resistance : ℝ),
      actual_current_speed ≥ current_speed_min →
      actual_current_speed ≤ current_speed_max →
      actual_wind_resistance ≥ wind_resistance_min →
      actual_wind_resistance ≤ wind_resistance_max →
      speed_against_current ≤ speed_with_current - actual_current_speed - 
        actual_wind_resistance * (speed_with_current - actual_current_speed)) :=
by sorry

end min_speed_against_current_l1045_104543


namespace cyclic_inequality_l1045_104512

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) * Real.sqrt ((y + z) * (z + x)) +
  (y + z) * Real.sqrt ((z + x) * (x + y)) +
  (z + x) * Real.sqrt ((x + y) * (y + z)) ≥
  4 * (x * y + y * z + z * x) :=
by sorry

end cyclic_inequality_l1045_104512


namespace product_of_zeros_range_l1045_104542

noncomputable section

def f (x : ℝ) : ℝ := 
  if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem product_of_zeros_range (m : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0) :
  ∃ p : ℝ, p < Real.sqrt (Real.exp 1) ∧ 
    ∀ q : ℝ, q < p → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0 ∧ x₁ * x₂ = q :=
sorry

end

end product_of_zeros_range_l1045_104542


namespace trailing_zeroes_of_six_factorial_l1045_104513

-- Define the function z(n) that counts trailing zeroes in n!
def z (n : ℕ) : ℕ := 
  (n / 5) + (n / 25) + (n / 125)

-- State the theorem
theorem trailing_zeroes_of_six_factorial : z (z 6) = 0 := by
  sorry

end trailing_zeroes_of_six_factorial_l1045_104513


namespace probability_nine_heads_in_twelve_flips_l1045_104548

def coin_flips : ℕ := 12
def heads_count : ℕ := 9

-- Define the probability of getting exactly k heads in n flips of a fair coin
def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_nine_heads_in_twelve_flips :
  probability_k_heads coin_flips heads_count = 55 / 1024 := by
  sorry

end probability_nine_heads_in_twelve_flips_l1045_104548


namespace quadratic_function_m_equals_one_l1045_104591

/-- A quadratic function passing through a point with specific x-range constraints -/
def QuadraticFunction (a b m t : ℝ) : Prop :=
  a ≠ 0 ∧
  2 = a * m^2 - b * m ∧
  ∀ x, (a * x^2 - b * x ≥ -1) → (x ≤ t - 1 ∨ x ≥ -3 - t)

/-- The theorem stating that m must equal 1 given the conditions -/
theorem quadratic_function_m_equals_one (a b m t : ℝ) :
  QuadraticFunction a b m t → m = 1 := by
  sorry

end quadratic_function_m_equals_one_l1045_104591


namespace inkblot_area_bound_l1045_104538

/-- Represents an inkblot on a square sheet of paper -/
structure Inkblot where
  area : ℝ
  x_extent : ℝ
  y_extent : ℝ

/-- The theorem stating that the total area of inkblots does not exceed the side length of the square paper -/
theorem inkblot_area_bound (a : ℝ) (inkblots : List Inkblot) : a > 0 →
  (∀ i ∈ inkblots, i.area ≤ 1) →
  (∀ i ∈ inkblots, i.x_extent ≤ a ∧ i.y_extent ≤ a) →
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ a → (inkblots.filter (fun i => i.x_extent > x)).length ≤ 1) →
  (∀ y : ℝ, y ≥ 0 ∧ y ≤ a → (inkblots.filter (fun i => i.y_extent > y)).length ≤ 1) →
  (inkblots.map (fun i => i.area)).sum ≤ a :=
by sorry

end inkblot_area_bound_l1045_104538
