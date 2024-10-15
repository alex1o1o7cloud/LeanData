import Mathlib

namespace NUMINAMATH_CALUDE_cell_phone_providers_l612_61242

theorem cell_phone_providers (n : ℕ) (k : ℕ) : n = 25 ∧ k = 4 → (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_providers_l612_61242


namespace NUMINAMATH_CALUDE_unique_number_property_l612_61205

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l612_61205


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l612_61253

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 / (x - 3) = 1 / x) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l612_61253


namespace NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l612_61295

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 18*y = k

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b h₁ h₂ : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), curve_equation x y k ↔ (x - h₁)^2 / a^2 + (y - h₂)^2 / b^2 = 1

/-- Theorem stating the condition for the curve to be a non-degenerate ellipse -/
theorem non_degenerate_ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -9 := by sorry

end NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l612_61295


namespace NUMINAMATH_CALUDE_unique_solution_l612_61220

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l612_61220


namespace NUMINAMATH_CALUDE_s_upper_bound_l612_61239

/-- Represents a triangle with side lengths p, q, r -/
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ
  h_positive : 0 < p ∧ 0 < q ∧ 0 < r
  h_inequality : p ≤ r ∧ r ≤ q
  h_triangle : p + r > q ∧ q + r > p ∧ p + q > r
  h_ratio : p / (q + r) = r / (p + q)

/-- Represents a point inside the triangle -/
structure InnerPoint (t : Triangle) where
  x : ℝ
  y : ℝ
  z : ℝ
  h_inside : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z < t.p + t.q + t.r

/-- The sum of distances from inner point to sides -/
def s (t : Triangle) (p : InnerPoint t) : ℝ := p.x + p.y + p.z

/-- The theorem to be proved -/
theorem s_upper_bound (t : Triangle) (p : InnerPoint t) : s t p ≤ 3 * t.p := by sorry

end NUMINAMATH_CALUDE_s_upper_bound_l612_61239


namespace NUMINAMATH_CALUDE_linda_cookies_theorem_l612_61285

/-- The number of batches Linda needs to bake to have enough cookies for her classmates -/
def batches_needed (num_classmates : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
  (choc_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : ℕ :=
  let total_cookies_needed := num_classmates * cookies_per_student
  let cookies_made := (choc_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let cookies_left_to_make := total_cookies_needed - cookies_made
  (cookies_left_to_make + cookies_per_batch - 1) / cookies_per_batch

theorem linda_cookies_theorem : 
  batches_needed 24 10 48 2 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linda_cookies_theorem_l612_61285


namespace NUMINAMATH_CALUDE_last_flip_heads_prob_2010_l612_61273

/-- A coin that comes up the same as the last flip 2/3 of the time and opposite 1/3 of the time -/
structure BiasedCoin where
  same_prob : ℚ
  diff_prob : ℚ
  prob_sum_one : same_prob + diff_prob = 1
  same_prob_val : same_prob = 2/3
  diff_prob_val : diff_prob = 1/3

/-- The probability of the last flip being heads after n flips, given the first flip was heads -/
def last_flip_heads_prob (coin : BiasedCoin) (n : ℕ) : ℚ :=
  (3^n + 1) / (2 * 3^n)

/-- The theorem statement -/
theorem last_flip_heads_prob_2010 (coin : BiasedCoin) :
  last_flip_heads_prob coin 2010 = (3^2010 + 1) / (2 * 3^2010) := by
  sorry

end NUMINAMATH_CALUDE_last_flip_heads_prob_2010_l612_61273


namespace NUMINAMATH_CALUDE_hyperbola_I_equation_equilateral_hyperbola_equation_l612_61276

-- Part I
def hyperbola_I (x y : ℝ) : Prop :=
  y^2 / 36 - x^2 / 28 = 1

theorem hyperbola_I_equation
  (foci_on_y_axis : True)
  (focal_distance : ℝ)
  (h_focal_distance : focal_distance = 16)
  (eccentricity : ℝ)
  (h_eccentricity : eccentricity = 4/3) :
  ∃ (x y : ℝ), hyperbola_I x y :=
sorry

-- Part II
def equilateral_hyperbola (x y : ℝ) : Prop :=
  x^2 / 18 - y^2 / 18 = 1

theorem equilateral_hyperbola_equation
  (is_equilateral : True)
  (focus : ℝ × ℝ)
  (h_focus : focus = (-6, 0)) :
  ∃ (x y : ℝ), equilateral_hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_I_equation_equilateral_hyperbola_equation_l612_61276


namespace NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_three_to_m_l612_61202

def m : ℕ := 2011^2 + 2^2011

theorem units_digit_of_m_cubed_plus_three_to_m (m : ℕ := 2011^2 + 2^2011) : 
  (m^3 + 3^m) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_three_to_m_l612_61202


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l612_61230

-- Define the sample space
def Ω : Type := Fin 3 → Bool

-- Define the events
def A (ω : Ω) : Prop := ∀ i, ω i = true
def B (ω : Ω) : Prop := ∀ i, ω i = false
def C (ω : Ω) : Prop := ∃ i j, ω i ≠ ω j

-- Theorem statement
theorem events_mutually_exclusive :
  (∀ ω, ¬(A ω ∧ B ω)) ∧
  (∀ ω, ¬(A ω ∧ C ω)) ∧
  (∀ ω, ¬(B ω ∧ C ω)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l612_61230


namespace NUMINAMATH_CALUDE_det_2CD_l612_61278

theorem det_2CD (C D : Matrix (Fin 3) (Fin 3) ℝ) 
  (hC : Matrix.det C = 3)
  (hD : Matrix.det D = 8) :
  Matrix.det (2 • (C * D)) = 192 := by
  sorry

end NUMINAMATH_CALUDE_det_2CD_l612_61278


namespace NUMINAMATH_CALUDE_problem_statement_l612_61207

theorem problem_statement (a b c : ℝ) :
  (∀ c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l612_61207


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l612_61223

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l612_61223


namespace NUMINAMATH_CALUDE_football_equipment_cost_l612_61200

/-- Given the cost of football equipment, prove the total cost relation -/
theorem football_equipment_cost (x : ℝ) 
  (h1 : x + x = 2 * x)           -- Shorts + T-shirt = 2x
  (h2 : x + 4 * x = 5 * x)       -- Shorts + boots = 5x
  (h3 : x + 2 * x = 3 * x)       -- Shorts + shin guards = 3x
  : x + x + 4 * x + 2 * x = 8 * x := by
  sorry


end NUMINAMATH_CALUDE_football_equipment_cost_l612_61200


namespace NUMINAMATH_CALUDE_salary_change_percentage_l612_61275

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * 1.5
  let final_salary := increased_salary * 0.9
  (final_salary - initial_salary) / initial_salary * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l612_61275


namespace NUMINAMATH_CALUDE_compound_carbon_atoms_l612_61214

/-- Represents a chemical compound --/
structure Compound where
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements --/
def atomic_weight : String → ℕ
  | "C" => 12
  | "H" => 1
  | "O" => 16
  | _ => 0

/-- Calculate the number of Carbon atoms in a compound --/
def carbon_atoms (c : Compound) : ℕ :=
  (c.molecular_weight - (c.hydrogen * atomic_weight "H" + c.oxygen * atomic_weight "O")) / atomic_weight "C"

/-- Theorem: The given compound has 4 Carbon atoms --/
theorem compound_carbon_atoms :
  let c : Compound := { hydrogen := 8, oxygen := 2, molecular_weight := 88 }
  carbon_atoms c = 4 := by
  sorry

end NUMINAMATH_CALUDE_compound_carbon_atoms_l612_61214


namespace NUMINAMATH_CALUDE_certain_number_proof_l612_61222

theorem certain_number_proof : ∃! x : ℚ, x / 4 + 3 = 5 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l612_61222


namespace NUMINAMATH_CALUDE_geometric_sequence_n_l612_61292

/-- For a geometric sequence {a_n} with a₁ = 9/8, q = 2/3, and aₙ = 1/3, n = 4 -/
theorem geometric_sequence_n (a : ℕ → ℚ) :
  (∀ k, a (k + 1) = a k * (2/3)) →  -- geometric sequence condition
  a 1 = 9/8 →                      -- first term condition
  (∃ n, a n = 1/3) →               -- nth term condition
  ∃ n, n = 4 ∧ a n = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_n_l612_61292


namespace NUMINAMATH_CALUDE_color_selection_ways_l612_61298

def total_colors : ℕ := 10
def colors_to_choose : ℕ := 3
def remaining_colors : ℕ := total_colors - 1  -- Subtracting blue

theorem color_selection_ways :
  (total_colors.choose colors_to_choose) - (remaining_colors.choose (colors_to_choose - 1)) =
  remaining_colors.choose (colors_to_choose - 1) := by
  sorry

end NUMINAMATH_CALUDE_color_selection_ways_l612_61298


namespace NUMINAMATH_CALUDE_equals_2022_l612_61262

theorem equals_2022 : 1 - (-2021) = 2022 := by
  sorry

#check equals_2022

end NUMINAMATH_CALUDE_equals_2022_l612_61262


namespace NUMINAMATH_CALUDE_five_athletes_three_events_l612_61283

/-- The number of different ways athletes can win championships in events -/
def championship_ways (num_athletes : ℕ) (num_events : ℕ) : ℕ :=
  num_athletes ^ num_events

/-- Theorem: 5 athletes winning 3 events results in 5^3 different ways -/
theorem five_athletes_three_events : 
  championship_ways 5 3 = 5^3 := by
  sorry

end NUMINAMATH_CALUDE_five_athletes_three_events_l612_61283


namespace NUMINAMATH_CALUDE_min_marked_cells_l612_61233

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece on the board -/
inductive LShape
  | makeL : Fin 2 → Fin 2 → LShape

/-- Checks if an L-shape touches a marked cell on the board -/
def touchesMarked (b : Board m n) (l : LShape) : Bool :=
  sorry

/-- Checks if a marking strategy satisfies the condition for all L-shape placements -/
def validMarking (b : Board m n) : Prop :=
  ∀ l : LShape, touchesMarked b l = true

/-- Counts the number of marked cells on the board -/
def countMarked (b : Board m n) : ℕ :=
  sorry

/-- The main theorem stating that 50 is the smallest number of cells to be marked -/
theorem min_marked_cells :
  ∃ (b : Board 10 11), validMarking b ∧ countMarked b = 50 ∧
  ∀ (b' : Board 10 11), validMarking b' → countMarked b' ≥ 50 :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_l612_61233


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l612_61264

/-- For the quadratic x^2 + 2200x + 4200, when written in the form (x+b)^2 + c, c/b = -1096 -/
theorem quadratic_form_ratio : ∃ (b c : ℝ), 
  (∀ x, x^2 + 2200*x + 4200 = (x + b)^2 + c) ∧ 
  c / b = -1096 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l612_61264


namespace NUMINAMATH_CALUDE_unique_solution_l612_61260

/-- Represents the work information for a worker -/
structure WorkerInfo where
  days : ℕ
  totalPay : ℕ
  dailyWage : ℚ

/-- Verifies if the given work information satisfies the problem conditions -/
def satisfiesConditions (a b : WorkerInfo) : Prop :=
  b.days = a.days - 3 ∧
  a.totalPay = 30 ∧
  b.totalPay = 14 ∧
  a.dailyWage = a.totalPay / a.days ∧
  b.dailyWage = b.totalPay / b.days ∧
  (a.days - 2) * a.dailyWage = (b.days + 5) * b.dailyWage

/-- The main theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (a b : WorkerInfo),
    satisfiesConditions a b ∧
    a.days = 10 ∧
    b.days = 7 ∧
    a.dailyWage = 3 ∧
    b.dailyWage = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l612_61260


namespace NUMINAMATH_CALUDE_potato_cost_for_group_l612_61241

/-- The cost of potatoes for a group, given the number of people, amount each person eats,
    bag size, and cost per bag. -/
def potatoCost (people : ℕ) (poundsPerPerson : ℚ) (bagSize : ℕ) (costPerBag : ℚ) : ℚ :=
  let totalPounds : ℚ := people * poundsPerPerson
  let bagsNeeded : ℕ := (totalPounds / bagSize).ceil.toNat
  bagsNeeded * costPerBag

/-- Theorem stating that the cost of potatoes for 40 people, where each person eats 1.5 pounds,
    and a 20-pound bag costs $5, is $15. -/
theorem potato_cost_for_group : potatoCost 40 (3/2) 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_potato_cost_for_group_l612_61241


namespace NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l612_61274

theorem smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11 :
  ∃ w : ℕ, w > 0 ∧ w % 13 = 0 ∧ (w + 3) % 11 = 0 ∧
  ∀ x : ℕ, x > 0 ∧ x % 13 = 0 ∧ (x + 3) % 11 = 0 → w ≤ x :=
by
  use 52
  sorry

end NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l612_61274


namespace NUMINAMATH_CALUDE_truck_rental_theorem_l612_61251

-- Define the capacities of small and large trucks
def small_truck_capacity : ℕ := 300
def large_truck_capacity : ℕ := 400

-- Define the conditions from the problem
axiom condition1 : 2 * small_truck_capacity + 3 * large_truck_capacity = 1800
axiom condition2 : 3 * small_truck_capacity + 4 * large_truck_capacity = 2500

-- Define the total items to be transported
def total_items : ℕ := 3100

-- Define a rental plan as a pair of natural numbers (small trucks, large trucks)
def RentalPlan := ℕ × ℕ

-- Define a function to check if a rental plan is valid
def is_valid_plan (plan : RentalPlan) : Prop :=
  plan.1 * small_truck_capacity + plan.2 * large_truck_capacity = total_items

-- Define the set of all valid rental plans
def valid_plans : Set RentalPlan :=
  {plan | is_valid_plan plan}

-- Theorem stating the main result
theorem truck_rental_theorem :
  (small_truck_capacity = 300 ∧ large_truck_capacity = 400) ∧
  (valid_plans = {(9, 1), (5, 4), (1, 7)}) := by
  sorry


end NUMINAMATH_CALUDE_truck_rental_theorem_l612_61251


namespace NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l612_61272

theorem probability_even_sum_two_wheels : 
  let wheel1_total := 6
  let wheel1_even := 3
  let wheel2_total := 4
  let wheel2_even := 3
  let prob_wheel1_even := wheel1_even / wheel1_total
  let prob_wheel1_odd := 1 - prob_wheel1_even
  let prob_wheel2_even := wheel2_even / wheel2_total
  let prob_wheel2_odd := 1 - prob_wheel2_even
  let prob_both_even := prob_wheel1_even * prob_wheel2_even
  let prob_both_odd := prob_wheel1_odd * prob_wheel2_odd
  prob_both_even + prob_both_odd = 1/2
:= by sorry

end NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l612_61272


namespace NUMINAMATH_CALUDE_volume_P_5_l612_61257

/-- Represents the volume of the dodecahedron after i iterations -/
def P (i : ℕ) : ℚ :=
  sorry

/-- The height of the tetrahedra at step i -/
def r (i : ℕ) : ℚ :=
  (1 / 2) ^ i

/-- The initial dodecahedron has volume 1 -/
axiom P_0 : P 0 = 1

/-- The recursive definition of P(i+1) based on P(i) and r(i) -/
axiom P_step (i : ℕ) : P (i + 1) = P i + 6 * (1 / 3) * (r i)^3

/-- The main theorem: the volume of P₅ is 8929/4096 -/
theorem volume_P_5 : P 5 = 8929 / 4096 :=
  sorry

end NUMINAMATH_CALUDE_volume_P_5_l612_61257


namespace NUMINAMATH_CALUDE_smallest_positive_solution_is_18_l612_61203

theorem smallest_positive_solution_is_18 : 
  let f : ℝ → ℝ := fun t => -t^2 + 14*t + 40
  ∃ t : ℝ, t > 0 ∧ f t = 94 ∧ ∀ s : ℝ, s > 0 ∧ f s = 94 → t ≤ s → t = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_is_18_l612_61203


namespace NUMINAMATH_CALUDE_shortest_path_between_circles_l612_61252

theorem shortest_path_between_circles (center_distance : Real) 
  (radius_large : Real) (radius_small : Real) : Real :=
by
  -- Define the conditions
  have h1 : center_distance = 51 := by sorry
  have h2 : radius_large = 12 := by sorry
  have h3 : radius_small = 7 := by sorry

  -- Calculate the length of the external tangent
  let total_distance := center_distance + radius_large + radius_small
  let tangent_length := Real.sqrt (total_distance^2 - radius_large^2)

  -- Prove that the tangent length is 69 feet
  have h4 : tangent_length = 69 := by sorry

  -- Return the result
  exact tangent_length

end NUMINAMATH_CALUDE_shortest_path_between_circles_l612_61252


namespace NUMINAMATH_CALUDE_product_of_sums_inequality_l612_61210

theorem product_of_sums_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (x + 1) * (y + 1) ≥ 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_product_of_sums_inequality_l612_61210


namespace NUMINAMATH_CALUDE_remainder_problem_l612_61282

theorem remainder_problem (x : ℤ) : 
  (4 * x) % 7 = 6 → x % 7 = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l612_61282


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l612_61235

/-- The total weight of Marco's and his dad's strawberries is 23 pounds. -/
theorem strawberry_weight_sum : 
  let marco_weight : ℕ := 14
  let dad_weight : ℕ := 9
  marco_weight + dad_weight = 23 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l612_61235


namespace NUMINAMATH_CALUDE_total_stuffed_animals_l612_61250

def stuffed_animals (mckenna kenley tenly : ℕ) : Prop :=
  (kenley = 2 * mckenna) ∧ 
  (tenly = kenley + 5) ∧ 
  (mckenna + kenley + tenly = 175)

theorem total_stuffed_animals :
  ∃ (mckenna kenley tenly : ℕ), 
    mckenna = 34 ∧ 
    stuffed_animals mckenna kenley tenly :=
by
  sorry

end NUMINAMATH_CALUDE_total_stuffed_animals_l612_61250


namespace NUMINAMATH_CALUDE_nondecreasing_function_l612_61254

-- Define the property that a sequence is nondecreasing
def IsNondecreasingSeq (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n ≤ m → s n ≤ s m

-- State the theorem
theorem nondecreasing_function 
  (f : ℝ → ℝ) 
  (hf_dom : ∀ x, 0 < x → f x ≠ 0) 
  (hf_cont : Continuous f) 
  (h_seq : ∀ x > 0, IsNondecreasingSeq (fun n ↦ f (n * x))) : 
  ∀ x y, 0 < x → 0 < y → x ≤ y → f x ≤ f y :=
sorry

end NUMINAMATH_CALUDE_nondecreasing_function_l612_61254


namespace NUMINAMATH_CALUDE_clockHandsOpposite_eq_48_l612_61287

/-- The number of times clock hands are in a straight line but opposite in direction in a day -/
def clockHandsOpposite : ℕ :=
  let hoursOnClockFace : ℕ := 12
  let hoursInDay : ℕ := 24
  let occurrencesPerHour : ℕ := 2
  hoursInDay * occurrencesPerHour

/-- Theorem stating that clock hands are in a straight line but opposite in direction 48 times a day -/
theorem clockHandsOpposite_eq_48 : clockHandsOpposite = 48 := by
  sorry

end NUMINAMATH_CALUDE_clockHandsOpposite_eq_48_l612_61287


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l612_61236

theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℝ) 
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_width = 30) :
  carol_length * carol_width = jordan_width * (120 / jordan_width) := by
  sorry

#check jordan_rectangle_length

end NUMINAMATH_CALUDE_jordan_rectangle_length_l612_61236


namespace NUMINAMATH_CALUDE_max_value_expression_l612_61204

theorem max_value_expression (a b c d : ℝ) 
  (ha : -5.5 ≤ a ∧ a ≤ 5.5)
  (hb : -5.5 ≤ b ∧ b ≤ 5.5)
  (hc : -5.5 ≤ c ∧ c ≤ 5.5)
  (hd : -5.5 ≤ d ∧ d ≤ 5.5) :
  (∀ a' b' c' d' : ℝ, 
    -5.5 ≤ a' ∧ a' ≤ 5.5 →
    -5.5 ≤ b' ∧ b' ≤ 5.5 →
    -5.5 ≤ c' ∧ c' ≤ 5.5 →
    -5.5 ≤ d' ∧ d' ≤ 5.5 →
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ 132) ∧
  (∃ a' b' c' d' : ℝ, 
    -5.5 ≤ a' ∧ a' ≤ 5.5 ∧
    -5.5 ≤ b' ∧ b' ≤ 5.5 ∧
    -5.5 ≤ c' ∧ c' ≤ 5.5 ∧
    -5.5 ≤ d' ∧ d' ≤ 5.5 ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 132) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l612_61204


namespace NUMINAMATH_CALUDE_absolute_value_condition_l612_61281

theorem absolute_value_condition (a : ℝ) : 
  (a ≤ 0 → |a - 2| ≥ 1) ∧ 
  ¬(|a - 2| ≥ 1 → a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_condition_l612_61281


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_two_fourth_quadrant_implies_m_lt_neg_two_m_eq_two_implies_sum_of_parts_l612_61206

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m - 1) * (m + 2) + (m - 1) * Complex.I

-- Theorem 1: If z is a pure imaginary number, then m = -2
theorem pure_imaginary_implies_m_eq_neg_two (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by sorry

-- Theorem 2: If z is in the fourth quadrant, then m < -2
theorem fourth_quadrant_implies_m_lt_neg_two (m : ℝ) :
  (z m).re > 0 ∧ (z m).im < 0 → m < -2 := by sorry

-- Theorem 3: If m = 2, then (z+i)/(z-1) = a + bi where a + b = 8/5
theorem m_eq_two_implies_sum_of_parts (m : ℝ) :
  m = 2 →
  ∃ a b : ℝ, (z m + Complex.I) / (z m - 1) = a + b * Complex.I ∧ a + b = 8/5 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_two_fourth_quadrant_implies_m_lt_neg_two_m_eq_two_implies_sum_of_parts_l612_61206


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l612_61266

theorem product_remainder_mod_five :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l612_61266


namespace NUMINAMATH_CALUDE_intersection_implies_a_range_l612_61219

/-- Given two functions f and g, where f(x) = ax and g(x) = ln x, 
    if their graphs intersect at two different points in (0, +∞),
    then 0 < a < 1/e. -/
theorem intersection_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ 
   a * x₁ = Real.log x₁ ∧ a * x₂ = Real.log x₂) →
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_range_l612_61219


namespace NUMINAMATH_CALUDE_base4_to_base10_conversion_l612_61248

/-- Converts a base 4 number represented as a list of digits to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number -/
def base4Number : List Nat := [2, 1, 0, 1, 2]

theorem base4_to_base10_conversion :
  base4ToBase10 base4Number = 582 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_base10_conversion_l612_61248


namespace NUMINAMATH_CALUDE_peach_trees_count_l612_61229

/-- The number of peach trees in an orchard. -/
def number_of_peach_trees (apple_trees : ℕ) (apple_yield : ℕ) (peach_yield : ℕ) (total_yield : ℕ) : ℕ :=
  (total_yield - apple_trees * apple_yield) / peach_yield

/-- Theorem stating the number of peach trees in the orchard. -/
theorem peach_trees_count : number_of_peach_trees 30 150 65 7425 = 45 := by
  sorry

end NUMINAMATH_CALUDE_peach_trees_count_l612_61229


namespace NUMINAMATH_CALUDE_power_of_two_equation_solution_l612_61269

theorem power_of_two_equation_solution : ∃ k : ℕ, 
  2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_solution_l612_61269


namespace NUMINAMATH_CALUDE_egg_price_calculation_l612_61212

/-- The price of a dozen eggs given the number of chickens, eggs laid per day, and total earnings --/
def price_per_dozen (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (total_earnings : ℕ) : ℚ :=
  let total_days : ℕ := 28  -- 4 weeks
  let total_eggs : ℕ := num_chickens * eggs_per_chicken_per_day * total_days
  let total_dozens : ℕ := total_eggs / 12
  (total_earnings : ℚ) / total_dozens

theorem egg_price_calculation :
  price_per_dozen 8 3 280 = 5 := by
  sorry

end NUMINAMATH_CALUDE_egg_price_calculation_l612_61212


namespace NUMINAMATH_CALUDE_gcd_of_squares_l612_61280

theorem gcd_of_squares : Nat.gcd (130^2 + 251^2 + 372^2) (129^2 + 250^2 + 373^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l612_61280


namespace NUMINAMATH_CALUDE_absolute_value_five_l612_61216

theorem absolute_value_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_five_l612_61216


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l612_61263

/-- The sum of the series Σ(k=1 to ∞) 3^k / (9^k - 1) is equal to 1/2 -/
theorem series_sum_equals_half :
  ∑' k, (3 : ℝ)^k / (9^k - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l612_61263


namespace NUMINAMATH_CALUDE_stream_rate_calculation_l612_61277

/-- The speed of the man rowing in still water (in kmph) -/
def still_water_speed : ℝ := 36

/-- The ratio of time taken to row upstream vs downstream -/
def upstream_downstream_ratio : ℝ := 3

/-- The rate of the stream (in kmph) -/
def stream_rate : ℝ := 18

theorem stream_rate_calculation :
  let d : ℝ := 1  -- Arbitrary distance
  let downstream_time := d / (still_water_speed + stream_rate)
  let upstream_time := d / (still_water_speed - stream_rate)
  upstream_time = upstream_downstream_ratio * downstream_time →
  stream_rate = 18 := by
sorry

end NUMINAMATH_CALUDE_stream_rate_calculation_l612_61277


namespace NUMINAMATH_CALUDE_pattern_properties_l612_61271

/-- Represents a figure in the pattern -/
structure Figure where
  n : ℕ

/-- Number of squares in a figure -/
def num_squares (f : Figure) : ℕ :=
  3 + 2 * (f.n - 1)

/-- Perimeter of a figure in cm -/
def perimeter (f : Figure) : ℕ :=
  8 + 2 * (f.n - 1)

theorem pattern_properties :
  ∀ (f : Figure),
    (num_squares f = 3 + 2 * (f.n - 1)) ∧
    (perimeter f = 8 + 2 * (f.n - 1)) ∧
    (perimeter ⟨16⟩ = 38) ∧
    ((perimeter ⟨29⟩ : ℚ) / (perimeter ⟨85⟩ : ℚ) = 4 / 11) :=
by sorry

end NUMINAMATH_CALUDE_pattern_properties_l612_61271


namespace NUMINAMATH_CALUDE_function_equation_solution_l612_61256

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) →
  (∀ x : ℝ, f x = (1/3) * (x^2 + 2*x - 1)) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l612_61256


namespace NUMINAMATH_CALUDE_angle_terminal_side_l612_61297

/-- Given that the terminal side of angle α passes through point (a, 1) and tan α = -1/2, prove that a = -2 -/
theorem angle_terminal_side (α : Real) (a : Real) : 
  (∃ (x y : Real), x = a ∧ y = 1 ∧ Real.tan α = y / x) → 
  Real.tan α = -1/2 → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l612_61297


namespace NUMINAMATH_CALUDE_sequence_properties_l612_61247

-- Define the sequence S_n
def S (n : ℕ+) : ℚ := n^2 + n

-- Define the sequence a_n
def a (n : ℕ+) : ℚ := 2 * n

-- Define the sequence T_n
def T (n : ℕ+) : ℚ := (1 / 2) * (1 - 1 / (n + 1))

theorem sequence_properties :
  (∀ n : ℕ+, S n = n^2 + n) →
  (∀ n : ℕ+, a n = 2 * n) ∧
  (∀ n : ℕ+, T n = (1 / 2) * (1 - 1 / (n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l612_61247


namespace NUMINAMATH_CALUDE_students_who_just_passed_l612_61261

theorem students_who_just_passed
  (total_students : ℕ)
  (first_division_percent : ℚ)
  (second_division_percent : ℚ)
  (h1 : total_students = 300)
  (h2 : first_division_percent = 27 / 100)
  (h3 : second_division_percent = 54 / 100)
  (h4 : first_division_percent + second_division_percent < 1) :
  total_students - (total_students * (first_division_percent + second_division_percent)).floor = 57 :=
by sorry

end NUMINAMATH_CALUDE_students_who_just_passed_l612_61261


namespace NUMINAMATH_CALUDE_total_lives_in_game_game_lives_proof_l612_61225

theorem total_lives_in_game (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) : ℕ :=
  (initial_players + additional_players) * lives_per_player

theorem game_lives_proof :
  total_lives_in_game 4 5 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_in_game_game_lives_proof_l612_61225


namespace NUMINAMATH_CALUDE_jonas_tshirts_l612_61218

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe where
  socks : ℕ
  shoes : ℕ
  pants : ℕ
  tshirts : ℕ

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  2 * w.socks + 2 * w.shoes + 2 * w.pants + w.tshirts

/-- The theorem to prove -/
theorem jonas_tshirts : 
  ∀ w : Wardrobe, 
    w.socks = 20 → 
    w.shoes = 5 → 
    w.pants = 10 → 
    totalItems w + 2 * 35 = 2 * totalItems w → 
    w.tshirts = 70 := by
  sorry


end NUMINAMATH_CALUDE_jonas_tshirts_l612_61218


namespace NUMINAMATH_CALUDE_product_abc_equals_195_l612_61294

/-- Given the conditions on products of variables a, b, c, d, e, f,
    prove that a * b * c equals 195. -/
theorem product_abc_equals_195
  (h1 : b * c * d = 65)
  (h2 : c * d * e = 1000)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 3/4)
  : a * b * c = 195 := by
  sorry

end NUMINAMATH_CALUDE_product_abc_equals_195_l612_61294


namespace NUMINAMATH_CALUDE_average_weight_whole_class_l612_61232

theorem average_weight_whole_class 
  (students_a : ℕ) (students_b : ℕ) 
  (avg_weight_a : ℚ) (avg_weight_b : ℚ) :
  students_a = 40 →
  students_b = 20 →
  avg_weight_a = 50 →
  avg_weight_b = 40 →
  (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = 140 / 3 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_whole_class_l612_61232


namespace NUMINAMATH_CALUDE_compound_interest_principal_l612_61288

def simple_interest_rate : ℝ := 0.14
def compound_interest_rate : ℝ := 0.07
def simple_interest_time : ℝ := 6
def compound_interest_time : ℝ := 2
def simple_interest_amount : ℝ := 603.75

theorem compound_interest_principal (P_SI : ℝ) (P_CI : ℝ) 
  (h1 : P_SI * simple_interest_rate * simple_interest_time = simple_interest_amount)
  (h2 : P_SI * simple_interest_rate * simple_interest_time = 
        1/2 * (P_CI * ((1 + compound_interest_rate) ^ compound_interest_time - 1)))
  (h3 : P_SI = 603.75 / (simple_interest_rate * simple_interest_time)) :
  P_CI = 8333.33 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l612_61288


namespace NUMINAMATH_CALUDE_ellipse_condition_l612_61299

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 + 2*y^2 - 6*x + 24*y = k

/-- The condition for a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), curve_equation x y k ↔ ((x - c)^2 / a + (y - d)^2 / b = e)

/-- The theorem stating the condition for the curve to be a non-degenerate ellipse -/
theorem ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -81 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l612_61299


namespace NUMINAMATH_CALUDE_rare_card_cost_proof_l612_61224

/-- The cost of each rare card in Tom's deck -/
def rare_card_cost : ℝ := 1

/-- The number of rare cards in Tom's deck -/
def num_rare_cards : ℕ := 19

/-- The number of uncommon cards in Tom's deck -/
def num_uncommon_cards : ℕ := 11

/-- The number of common cards in Tom's deck -/
def num_common_cards : ℕ := 30

/-- The cost of each uncommon card -/
def uncommon_card_cost : ℝ := 0.50

/-- The cost of each common card -/
def common_card_cost : ℝ := 0.25

/-- The total cost of Tom's deck -/
def total_deck_cost : ℝ := 32

theorem rare_card_cost_proof :
  rare_card_cost * num_rare_cards +
  uncommon_card_cost * num_uncommon_cards +
  common_card_cost * num_common_cards = total_deck_cost :=
by sorry

end NUMINAMATH_CALUDE_rare_card_cost_proof_l612_61224


namespace NUMINAMATH_CALUDE_right_triangle_ratio_range_l612_61265

theorem right_triangle_ratio_range (a b c h : ℝ) :
  a > 0 → b > 0 →
  c = (a^2 + b^2).sqrt →
  h = (a * b) / c →
  1 < (c + 2 * h) / (a + b) ∧ (c + 2 * h) / (a + b) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_range_l612_61265


namespace NUMINAMATH_CALUDE_stones_for_hall_l612_61213

-- Define the hall dimensions in decimeters
def hall_length : ℕ := 360
def hall_width : ℕ := 150

-- Define the stone dimensions in decimeters
def stone_length : ℕ := 3
def stone_width : ℕ := 5

-- Define the function to calculate the number of stones required
def stones_required (hall_l hall_w stone_l stone_w : ℕ) : ℕ :=
  (hall_l * hall_w) / (stone_l * stone_w)

-- Theorem statement
theorem stones_for_hall :
  stones_required hall_length hall_width stone_length stone_width = 3600 := by
  sorry

end NUMINAMATH_CALUDE_stones_for_hall_l612_61213


namespace NUMINAMATH_CALUDE_utility_bill_total_l612_61279

def fifty_bill_value : ℕ := 50
def ten_bill_value : ℕ := 10
def fifty_bill_count : ℕ := 3
def ten_bill_count : ℕ := 2

theorem utility_bill_total : 
  fifty_bill_value * fifty_bill_count + ten_bill_value * ten_bill_count = 170 := by
  sorry

end NUMINAMATH_CALUDE_utility_bill_total_l612_61279


namespace NUMINAMATH_CALUDE_find_wrong_height_l612_61245

/-- Given a class of boys with an initially miscalculated average height and the correct average height after fixing one boy's height, find the wrongly written height of that boy. -/
theorem find_wrong_height (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (correct_avg : ℝ) 
    (hn : n = 35)
    (hi : initial_avg = 181)
    (ha : actual_height = 106)
    (hc : correct_avg = 179) :
    ∃ wrong_height : ℝ,
      wrong_height = n * initial_avg - (n * correct_avg - actual_height) :=
by sorry

end NUMINAMATH_CALUDE_find_wrong_height_l612_61245


namespace NUMINAMATH_CALUDE_total_distance_triangle_l612_61258

theorem total_distance_triangle (XZ XY : ℝ) (h1 : XZ = 5000) (h2 : XY = 5200) :
  let YZ := Real.sqrt (XY ^ 2 - XZ ^ 2)
  XZ + XY + YZ = 11628 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_triangle_l612_61258


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l612_61226

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![0, 5; 0, 10]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; 4, 3]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![10, 15; 20, 6]
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l612_61226


namespace NUMINAMATH_CALUDE_total_age_is_22_l612_61208

/-- Given three people A, B, and C with the following age relationships:
    - A is two years older than B
    - B is twice as old as C
    - B is 8 years old
    This theorem proves that the sum of their ages is 22 years. -/
theorem total_age_is_22 (a b c : ℕ) : 
  b = 8 → a = b + 2 → b = 2 * c → a + b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_age_is_22_l612_61208


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l612_61268

/-- Represents a trapezoid ABCD with given properties -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  AD_eq_BC : AD = BC
  O_interior : O_in_interior
  OT : ℝ

/-- The area of the isosceles trapezoid with the given properties -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ (t : IsoscelesTrapezoid),
    t.AB = 6 ∧ t.CD = 12 ∧ t.OT = 18 ∧
    trapezoid_area t = 54 + 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l612_61268


namespace NUMINAMATH_CALUDE_equation_linearity_l612_61291

/-- The equation (k^2 - 1)x^2 + (k + 1)x + (k - 7)y = k + 2 -/
def equation (k x y : ℝ) : Prop :=
  (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y = k + 2

/-- The equation is linear in one variable -/
def is_linear_one_var (k : ℝ) : Prop :=
  k^2 - 1 = 0 ∧ k + 1 = 0

/-- The equation is linear in two variables -/
def is_linear_two_var (k : ℝ) : Prop :=
  k^2 - 1 = 0 ∧ k + 1 ≠ 0

theorem equation_linearity :
  (is_linear_one_var (-1) ∧ is_linear_two_var 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_linearity_l612_61291


namespace NUMINAMATH_CALUDE_street_trees_count_l612_61217

theorem street_trees_count (road_length : ℕ) (interval : ℕ) : 
  road_length = 2575 → interval = 25 → (road_length / interval + 1 : ℕ) = 104 := by
  sorry

end NUMINAMATH_CALUDE_street_trees_count_l612_61217


namespace NUMINAMATH_CALUDE_root_implies_k_value_l612_61286

theorem root_implies_k_value (k : ℝ) : ((-3 : ℝ)^2 + (-3) - k = 0) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l612_61286


namespace NUMINAMATH_CALUDE_solve_bones_problem_l612_61234

def bones_problem (initial_bones final_bones : ℕ) : Prop :=
  let doubled_bones := 2 * initial_bones
  let stolen_bones := doubled_bones - final_bones
  stolen_bones = 2

theorem solve_bones_problem :
  bones_problem 4 6 := by sorry

end NUMINAMATH_CALUDE_solve_bones_problem_l612_61234


namespace NUMINAMATH_CALUDE_inequality_properties_l612_61255

theorem inequality_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1 / a < 1 / b) := by sorry

end NUMINAMATH_CALUDE_inequality_properties_l612_61255


namespace NUMINAMATH_CALUDE_james_coffee_consumption_l612_61209

/-- Proves that James bought 2 coffees per day before buying a coffee machine -/
theorem james_coffee_consumption
  (machine_cost : ℕ)
  (daily_making_cost : ℕ)
  (previous_coffee_cost : ℕ)
  (payoff_days : ℕ)
  (h1 : machine_cost = 180)
  (h2 : daily_making_cost = 3)
  (h3 : previous_coffee_cost = 4)
  (h4 : payoff_days = 36) :
  ∃ x : ℕ, x = 2 ∧ payoff_days * (previous_coffee_cost * x - daily_making_cost) = machine_cost :=
by sorry

end NUMINAMATH_CALUDE_james_coffee_consumption_l612_61209


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l612_61270

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 110 → 
  train_speed_kmh = 72 → 
  bridge_length = 142 → 
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l612_61270


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l612_61227

theorem cube_root_equation_solutions :
  ∀ x : ℝ, (x^(1/3) = 15 / (10 - x^(1/3))) ↔ (x = 125 ∨ x = 27) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l612_61227


namespace NUMINAMATH_CALUDE_parabola_transformation_transformed_vertex_l612_61290

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 4)^2 - 1

/-- Theorem stating that the transformed parabola is a shift of the original parabola -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 4) - 1 := by
  sorry

/-- Corollary showing the vertex of the transformed parabola -/
theorem transformed_vertex :
  ∃ x y : ℝ, x = 4 ∧ y = -1 ∧ ∀ t : ℝ, transformed_parabola t ≥ transformed_parabola x := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_transformed_vertex_l612_61290


namespace NUMINAMATH_CALUDE_tan_theta_value_l612_61246

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_theta_value (θ : Real) (h : ∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l612_61246


namespace NUMINAMATH_CALUDE_bacon_strips_for_fourteen_customers_l612_61293

/-- Breakfast plate configuration at a cafe -/
structure BreakfastPlate where
  eggs : ℕ
  bacon_multiplier : ℕ

/-- Calculate total bacon strips needed for multiple breakfast plates -/
def total_bacon_strips (plate : BreakfastPlate) (num_customers : ℕ) : ℕ :=
  num_customers * (plate.eggs * plate.bacon_multiplier)

/-- Theorem: The cook needs to fry 56 bacon strips for 14 customers -/
theorem bacon_strips_for_fourteen_customers :
  ∃ (plate : BreakfastPlate),
    plate.eggs = 2 ∧
    plate.bacon_multiplier = 2 ∧
    total_bacon_strips plate 14 = 56 := by
  sorry

end NUMINAMATH_CALUDE_bacon_strips_for_fourteen_customers_l612_61293


namespace NUMINAMATH_CALUDE_quiz_bowl_points_per_answer_l612_61211

/-- Represents the quiz bowl game structure and James' performance --/
structure QuizBowl where
  total_rounds : Nat
  questions_per_round : Nat
  bonus_points : Nat
  james_total_points : Nat
  james_missed_questions : Nat

/-- Calculates the points per correct answer in the quiz bowl --/
def points_per_correct_answer (qb : QuizBowl) : Nat :=
  let total_questions := qb.total_rounds * qb.questions_per_round
  let james_correct_answers := total_questions - qb.james_missed_questions
  let perfect_rounds := (james_correct_answers / qb.questions_per_round)
  let bonus_total := perfect_rounds * qb.bonus_points
  let points_from_answers := qb.james_total_points - bonus_total
  points_from_answers / james_correct_answers

/-- Theorem stating that given the specific conditions, the points per correct answer is 2 --/
theorem quiz_bowl_points_per_answer :
  let qb : QuizBowl := {
    total_rounds := 5,
    questions_per_round := 5,
    bonus_points := 4,
    james_total_points := 66,
    james_missed_questions := 1
  }
  points_per_correct_answer qb = 2 := by
  sorry

end NUMINAMATH_CALUDE_quiz_bowl_points_per_answer_l612_61211


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l612_61296

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) 
  (inradius : ℝ) 
  (h_perimeter : perimeter = 42) 
  (h_inradius : inradius = 5) : 
  inradius * (perimeter / 2) = 105 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l612_61296


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l612_61259

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (additional_purchase : ℝ) 
  (additional_decaf_percent : ℝ) 
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_purchase = 100)
  (h4 : additional_decaf_percent = 60) :
  let initial_decaf := initial_stock * (initial_decaf_percent / 100)
  let additional_decaf := additional_purchase * (additional_decaf_percent / 100)
  let total_decaf := initial_decaf + additional_decaf
  let total_stock := initial_stock + additional_purchase
  (total_decaf / total_stock) * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l612_61259


namespace NUMINAMATH_CALUDE_plain_pancakes_count_l612_61221

/-- Given a total of 67 pancakes, with 20 having blueberries and 24 having bananas,
    prove that there are 23 plain pancakes. -/
theorem plain_pancakes_count (total : ℕ) (blueberry : ℕ) (banana : ℕ) 
  (h1 : total = 67) 
  (h2 : blueberry = 20) 
  (h3 : banana = 24) :
  total - (blueberry + banana) = 23 := by
  sorry

#check plain_pancakes_count

end NUMINAMATH_CALUDE_plain_pancakes_count_l612_61221


namespace NUMINAMATH_CALUDE_system_solution_unique_l612_61284

theorem system_solution_unique (x y : ℝ) : 
  x + y = 5 ∧ 3 * x + y = 7 ↔ x = 1 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l612_61284


namespace NUMINAMATH_CALUDE_harmonic_progression_solutions_l612_61243

def is_harmonic_progression (a b c : ℕ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (1 / a + 1 / c = 2 / b)

def valid_harmonic_progression (a b c : ℕ) : Prop :=
  is_harmonic_progression a b c ∧ a < b ∧ b < c ∧ a = 20 ∧ c % b = 0

theorem harmonic_progression_solutions :
  {(b, c) : ℕ × ℕ | valid_harmonic_progression 20 b c} =
    {(30, 60), (35, 140), (36, 180), (38, 380), (39, 780)} := by
  sorry

end NUMINAMATH_CALUDE_harmonic_progression_solutions_l612_61243


namespace NUMINAMATH_CALUDE_equation_solution_l612_61231

theorem equation_solution : 
  ∃! x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 3) ∧ x = -1/19 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l612_61231


namespace NUMINAMATH_CALUDE_complement_M_in_U_l612_61215

def U : Set ℕ := {x | x < 5 ∧ x > 0}
def M : Set ℕ := {x | x ^ 2 - 5 * x + 6 = 0}

theorem complement_M_in_U : U \ M = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l612_61215


namespace NUMINAMATH_CALUDE_factorial_fraction_l612_61244

theorem factorial_fraction (N : ℕ) : 
  (Nat.factorial (N - 1) * (N^2 + N)) / Nat.factorial (N + 2) = 1 / (N + 2) :=
sorry

end NUMINAMATH_CALUDE_factorial_fraction_l612_61244


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_1812_l612_61240

/-- Calculates the cost of white washing a room with given dimensions and openings. -/
def whitewashingCost (length width height : ℝ) (doorHeight doorWidth : ℝ) 
  (windowHeight windowWidth : ℝ) (numWindows : ℕ) (ratePerSqFt : ℝ) : ℝ :=
  let wallArea := 2 * (length + width) * height
  let doorArea := doorHeight * doorWidth
  let windowArea := windowHeight * windowWidth * (numWindows : ℝ)
  let adjustedArea := wallArea - doorArea - windowArea
  adjustedArea * ratePerSqFt

/-- The cost of white washing the room is Rs. 1812. -/
theorem whitewashing_cost_is_1812 :
  whitewashingCost 25 15 12 6 3 4 3 3 2 = 1812 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_1812_l612_61240


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l612_61249

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l612_61249


namespace NUMINAMATH_CALUDE_max_d_value_l612_61201

def a (n : ℕ+) : ℕ := 101 + n.val ^ 2 + 3 * n.val

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∀ n : ℕ+, d n ≤ 4 ∧ ∃ m : ℕ+, d m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l612_61201


namespace NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l612_61267

/-- Represents a chessboard configuration -/
structure ChessBoard where
  size : Nat
  rooks : Nat

/-- Calculates the minimum number of attacking rook pairs -/
def minAttackingPairs (board : ChessBoard) : Nat :=
  sorry

/-- Theorem stating the minimum number of attacking rook pairs for a specific configuration -/
theorem min_attacking_pairs_8x8_16rooks :
  let board : ChessBoard := { size := 8, rooks := 16 }
  minAttackingPairs board = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l612_61267


namespace NUMINAMATH_CALUDE_sequence_characterization_l612_61228

theorem sequence_characterization (a : ℕ+ → ℝ) :
  (∀ m n : ℕ+, a (m + n) = a m + a n - (m * n : ℝ)) ∧
  (∀ m n : ℕ+, a (m * n) = (m ^ 2 : ℝ) * a n + (n ^ 2 : ℝ) * a m + 2 * a m * a n) →
  (∀ n : ℕ+, a n = -(n * (n - 1) : ℝ) / 2) ∨
  (∀ n : ℕ+, a n = -(n ^ 2 : ℝ) / 2) := by
sorry

end NUMINAMATH_CALUDE_sequence_characterization_l612_61228


namespace NUMINAMATH_CALUDE_grid_midpoint_theorem_l612_61289

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) :
  points.card = 5 →
  ∃ p q : ℤ × ℤ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    Even (p.1 + q.1) ∧ Even (p.2 + q.2) :=
by sorry

end NUMINAMATH_CALUDE_grid_midpoint_theorem_l612_61289


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l612_61238

theorem imaginary_part_of_complex_fraction (i : Complex) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (1 + i)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l612_61238


namespace NUMINAMATH_CALUDE_sin_270_degrees_l612_61237

theorem sin_270_degrees : Real.sin (270 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_270_degrees_l612_61237
