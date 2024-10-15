import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_l2669_266997

/-- The area of a triangle with base 2 and height 3 is 3 -/
theorem triangle_area : 
  let base : ℝ := 2
  let height : ℝ := 3
  let area := (base * height) / 2
  area = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2669_266997


namespace NUMINAMATH_CALUDE_rectangle_equation_l2669_266913

theorem rectangle_equation (x : ℝ) : 
  (∀ L W : ℝ, L * W = 864 ∧ L + W = 60 ∧ L = W + x) →
  (60 - x) / 2 * (60 + x) / 2 = 864 := by
sorry

end NUMINAMATH_CALUDE_rectangle_equation_l2669_266913


namespace NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2669_266978

noncomputable def f (a b x : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_of_even_increasing_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_incr : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  ∀ x, f a b (2 - x) > 0 ↔ x < 0 ∨ x > 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2669_266978


namespace NUMINAMATH_CALUDE_count_numbers_greater_than_three_l2669_266973

theorem count_numbers_greater_than_three : 
  let numbers : Finset ℝ := {0.8, 1/2, 0.9, 1/3}
  (numbers.filter (λ x => x > 3)).card = 0 := by
sorry

end NUMINAMATH_CALUDE_count_numbers_greater_than_three_l2669_266973


namespace NUMINAMATH_CALUDE_merchant_profit_l2669_266962

theorem merchant_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 20 → 
  discount_percent = 10 → 
  let marked_price := cost * (1 + markup_percent / 100)
  let final_price := marked_price * (1 - discount_percent / 100)
  let profit_percent := (final_price - cost) / cost * 100
  profit_percent = 8 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l2669_266962


namespace NUMINAMATH_CALUDE_parallel_line_length_l2669_266989

theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 18 → 
  (parallel_line / base)^2 = 1/2 → 
  parallel_line = 9 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l2669_266989


namespace NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l2669_266922

/-- Represents the value of 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The tourism revenue in billions of yuan -/
def tourism_revenue : ℝ := 12.41

theorem tourism_revenue_scientific_notation : 
  tourism_revenue * billion = 1.241 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l2669_266922


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l2669_266960

theorem cubic_equation_unique_solution :
  ∃! (x y : ℕ+), (y : ℤ)^3 = (x : ℤ)^3 + 8*(x : ℤ)^2 - 6*(x : ℤ) + 8 ∧ x = 9 ∧ y = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l2669_266960


namespace NUMINAMATH_CALUDE_bianca_candy_count_l2669_266901

def candy_problem (eaten : ℕ) (pieces_per_pile : ℕ) (num_piles : ℕ) : ℕ :=
  eaten + pieces_per_pile * num_piles

theorem bianca_candy_count : candy_problem 12 5 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_bianca_candy_count_l2669_266901


namespace NUMINAMATH_CALUDE_least_subtraction_for_common_remainder_l2669_266942

theorem least_subtraction_for_common_remainder (n : ℕ) : 
  (∃ (x : ℕ), x ≤ n ∧ 
   (642 - x) % 11 = 4 ∧ 
   (642 - x) % 13 = 4 ∧ 
   (642 - x) % 17 = 4) → 
  (∃ (x : ℕ), x ≤ n ∧ 
   (642 - x) % 11 = 4 ∧ 
   (642 - x) % 13 = 4 ∧ 
   (642 - x) % 17 = 4 ∧
   ∀ (y : ℕ), y < x → 
     ((642 - y) % 11 ≠ 4 ∨ 
      (642 - y) % 13 ≠ 4 ∨ 
      (642 - y) % 17 ≠ 4)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_common_remainder_l2669_266942


namespace NUMINAMATH_CALUDE_homework_problem_l2669_266918

theorem homework_problem (p t : ℕ) (h1 : p > 12) (h2 : t > 0) 
  (h3 : p * t = (p + 6) * (t - 3)) : p * t = 140 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l2669_266918


namespace NUMINAMATH_CALUDE_equation_solutions_l2669_266902

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 = x ↔ x = 0 ∨ x = 1/4) ∧
  (∀ x : ℝ, x^2 - 18*x + 1 = 0 ↔ x = 9 + 4*Real.sqrt 5 ∨ x = 9 - 4*Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2669_266902


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l2669_266984

theorem triangle_inequality_cube (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^3 + b^3 + 3*a*b*c > c^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l2669_266984


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2669_266941

theorem min_value_squared_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  x^2 + y^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2669_266941


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2669_266952

theorem solution_set_of_inequality (x : ℝ) :
  (x / (2 * x - 1) > 1) ↔ (1/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2669_266952


namespace NUMINAMATH_CALUDE_CD_length_approx_l2669_266916

/-- A quadrilateral with intersecting diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (BO : ℝ)
  (OD : ℝ)
  (AO : ℝ)
  (OC : ℝ)
  (AB : ℝ)

/-- The length of CD in the quadrilateral -/
def CD_length (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating the length of CD in the given quadrilateral -/
theorem CD_length_approx (q : Quadrilateral) 
  (h1 : q.BO = 3)
  (h2 : q.OD = 5)
  (h3 : q.AO = 7)
  (h4 : q.OC = 4)
  (h5 : q.AB = 5) :
  ∃ ε > 0, |CD_length q - 8.51| < ε :=
sorry

end NUMINAMATH_CALUDE_CD_length_approx_l2669_266916


namespace NUMINAMATH_CALUDE_troy_needs_ten_dollars_l2669_266900

/-- The amount of additional money Troy needs to buy a new computer -/
def additional_money_needed (new_computer_cost initial_savings old_computer_price : ℕ) : ℕ :=
  new_computer_cost - (initial_savings + old_computer_price)

/-- Theorem: Troy needs $10 more to buy the new computer -/
theorem troy_needs_ten_dollars : 
  additional_money_needed 80 50 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_troy_needs_ten_dollars_l2669_266900


namespace NUMINAMATH_CALUDE_sum_of_products_even_l2669_266936

/-- Represents a regular hexagon with natural numbers assigned to its vertices -/
structure Hexagon where
  vertices : Fin 6 → ℕ

/-- The sum of products of adjacent vertex pairs in a hexagon -/
def sum_of_products (h : Hexagon) : ℕ :=
  (h.vertices 0 * h.vertices 1) + (h.vertices 1 * h.vertices 2) +
  (h.vertices 2 * h.vertices 3) + (h.vertices 3 * h.vertices 4) +
  (h.vertices 4 * h.vertices 5) + (h.vertices 5 * h.vertices 0)

/-- A hexagon with opposite vertices having the same value -/
def opposite_same_hexagon (a b c : ℕ) : Hexagon :=
  { vertices := fun i => match i with
    | 0 | 3 => a
    | 1 | 4 => b
    | 2 | 5 => c }

theorem sum_of_products_even (a b c : ℕ) :
  Even (sum_of_products (opposite_same_hexagon a b c)) := by
  sorry

#check sum_of_products_even

end NUMINAMATH_CALUDE_sum_of_products_even_l2669_266936


namespace NUMINAMATH_CALUDE_square_side_length_when_area_equals_perimeter_l2669_266964

theorem square_side_length_when_area_equals_perimeter :
  ∃ (a : ℝ), a > 0 ∧ a^2 = 4*a := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_when_area_equals_perimeter_l2669_266964


namespace NUMINAMATH_CALUDE_max_fences_for_100_houses_prove_max_fences_199_l2669_266928

/-- Represents a village with houses and fences. -/
structure Village where
  num_houses : ℕ
  num_fences : ℕ

/-- Represents the process of combining houses within a fence. -/
def combine_houses (v : Village) : Village :=
  { num_houses := v.num_houses - 1
  , num_fences := v.num_fences - 2 }

/-- The maximum number of fences for a given number of houses. -/
def max_fences (n : ℕ) : ℕ :=
  2 * n - 1

/-- Theorem stating the maximum number of fences for 100 houses. -/
theorem max_fences_for_100_houses :
  ∃ (v : Village), v.num_houses = 100 ∧ v.num_fences = max_fences v.num_houses :=
by
  sorry

/-- Theorem proving that 199 is the maximum number of fences for 100 houses. -/
theorem prove_max_fences_199 :
  max_fences 100 = 199 :=
by
  sorry

end NUMINAMATH_CALUDE_max_fences_for_100_houses_prove_max_fences_199_l2669_266928


namespace NUMINAMATH_CALUDE_power_three_250_mod_13_l2669_266905

theorem power_three_250_mod_13 : 3^250 % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_three_250_mod_13_l2669_266905


namespace NUMINAMATH_CALUDE_students_tree_planting_l2669_266981

/-- The number of apple trees planted by students -/
def apple_trees : ℕ := 47

/-- The number of orange trees planted by students -/
def orange_trees : ℕ := 27

/-- The total number of trees planted by students -/
def total_trees : ℕ := apple_trees + orange_trees

theorem students_tree_planting : total_trees = 74 := by
  sorry

end NUMINAMATH_CALUDE_students_tree_planting_l2669_266981


namespace NUMINAMATH_CALUDE_correct_calculation_l2669_266958

theorem correct_calculation : (36 - 12) / (3 / 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2669_266958


namespace NUMINAMATH_CALUDE_letter_150_is_B_l2669_266954

def letter_sequence : ℕ → Char
  | n => match n % 4 with
    | 0 => 'A'
    | 1 => 'B'
    | 2 => 'C'
    | _ => 'D'

theorem letter_150_is_B : letter_sequence 149 = 'B' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_B_l2669_266954


namespace NUMINAMATH_CALUDE_smallest_value_3a_plus_2_l2669_266957

theorem smallest_value_3a_plus_2 (a : ℝ) (h : 4 * a^2 + 6 * a + 3 = 2) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (x : ℝ), (∃ (b : ℝ), 4 * b^2 + 6 * b + 3 = 2 ∧ x = 3 * b + 2) → x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_3a_plus_2_l2669_266957


namespace NUMINAMATH_CALUDE_sequence_existence_l2669_266915

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem sequence_existence
  (f : ℕ → ℕ) (h_inc : StrictlyIncreasing f) :
  (∃ y : ℕ → ℝ, (∀ n, y n > 0) ∧
    (∀ n m, n < m → y m < y n) ∧
    (∀ ε > 0, ∃ N, ∀ n ≥ N, y n < ε) ∧
    (∀ n, y n ≤ 2 * y (f n))) ∧
  (∀ x : ℕ → ℝ,
    (∀ n m, n < m → x m < x n) →
    (∀ ε > 0, ∃ N, ∀ n ≥ N, x n < ε) →
    ∃ y : ℕ → ℝ,
      (∀ n m, n < m → y m < y n) ∧
      (∀ ε > 0, ∃ N, ∀ n ≥ N, y n < ε) ∧
      (∀ n, x n ≤ y n ∧ y n ≤ 2 * y (f n))) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_l2669_266915


namespace NUMINAMATH_CALUDE_mary_lambs_count_l2669_266953

def lambs_problem (initial_lambs : ℕ) (mother_lambs : ℕ) (babies_per_lamb : ℕ) 
                  (traded_lambs : ℕ) (extra_lambs : ℕ) : Prop :=
  let new_babies := mother_lambs * babies_per_lamb
  let after_births := initial_lambs + new_babies
  let after_trade := after_births - traded_lambs
  let final_count := after_trade + extra_lambs
  final_count = 34

theorem mary_lambs_count : 
  lambs_problem 12 4 3 5 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_lambs_count_l2669_266953


namespace NUMINAMATH_CALUDE_cube_surface_area_l2669_266951

theorem cube_surface_area (volume : ℝ) (surface_area : ℝ) : 
  volume = 3375 → surface_area = 1350 → 
  (∃ (side : ℝ), volume = side^3 ∧ surface_area = 6 * side^2) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2669_266951


namespace NUMINAMATH_CALUDE_parabola_point_y_coord_l2669_266917

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y
  distance_to_focus : (x - 0)^2 + (y - 1)^2 = 2^2

/-- Theorem: The y-coordinate of a point on the parabola x^2 = 4y that is 2 units away from the focus (0, 1) is 1 -/
theorem parabola_point_y_coord (P : ParabolaPoint) : P.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_y_coord_l2669_266917


namespace NUMINAMATH_CALUDE_rectangle_reassembly_l2669_266965

/-- For any positive real numbers a and b, there exists a way to cut and reassemble
    a rectangle with dimensions a and b into a rectangle with one side equal to 1. -/
theorem rectangle_reassembly (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (c : ℝ), c > 0 ∧ a * b = c * 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_reassembly_l2669_266965


namespace NUMINAMATH_CALUDE_puzzle_solution_l2669_266931

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- The problem statement -/
theorem puzzle_solution 
  (EH OY AY OH : TwoDigitNumber)
  (h1 : EH.val = 4 * OY.val)
  (h2 : AY.val = 4 * OH.val) :
  EH.val + OY.val + AY.val + OH.val = 150 :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2669_266931


namespace NUMINAMATH_CALUDE_parallelogram_area_l2669_266974

/-- The area of a parallelogram with base 22 cm and height 21 cm is 462 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 22 → 
    height = 21 → 
    area = base * height → 
    area = 462 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2669_266974


namespace NUMINAMATH_CALUDE_parabola_max_ratio_l2669_266949

theorem parabola_max_ratio (p : ℝ) (h : p > 0) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 / 4 ∧
  ∀ (x y : ℝ), y^2 = 2*p*x →
    Real.sqrt (x^2 + y^2) / Real.sqrt ((x - p/6)^2 + y^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_ratio_l2669_266949


namespace NUMINAMATH_CALUDE_faster_train_distance_and_time_l2669_266932

/-- Represents the speed and distance of a train -/
structure Train where
  speed : ℝ
  distance : ℝ

/-- Proves the distance covered by a faster train and the time taken -/
theorem faster_train_distance_and_time 
  (old_train : Train)
  (new_train : Train)
  (speed_increase_percent : ℝ)
  (h1 : old_train.distance = 300)
  (h2 : new_train.speed = old_train.speed * (1 + speed_increase_percent))
  (h3 : speed_increase_percent = 0.3)
  (h4 : new_train.speed = 120) : 
  new_train.distance = 390 ∧ (new_train.distance / new_train.speed) = 3.25 := by
  sorry

#check faster_train_distance_and_time

end NUMINAMATH_CALUDE_faster_train_distance_and_time_l2669_266932


namespace NUMINAMATH_CALUDE_largest_common_value_proof_l2669_266907

/-- First arithmetic progression with initial term 4 and common difference 5 -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- Second arithmetic progression with initial term 5 and common difference 8 -/
def seq2 (m : ℕ) : ℕ := 5 + 8 * m

/-- The largest common value less than 1000 in both sequences -/
def largest_common_value : ℕ := 989

theorem largest_common_value_proof :
  (∃ n m : ℕ, seq1 n = largest_common_value ∧ seq2 m = largest_common_value) ∧
  (∀ k : ℕ, k < 1000 → (∃ n m : ℕ, seq1 n = k ∧ seq2 m = k) → k ≤ largest_common_value) :=
sorry

end NUMINAMATH_CALUDE_largest_common_value_proof_l2669_266907


namespace NUMINAMATH_CALUDE_ribbon_leftover_l2669_266975

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) 
  (h1 : total_ribbon = 18)
  (h2 : num_gifts = 6)
  (h3 : ribbon_per_gift = 2) : 
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_leftover_l2669_266975


namespace NUMINAMATH_CALUDE_succeeding_number_in_base_3_l2669_266927

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (3^i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  sorry  -- Implementation not provided as it's not needed for the statement

def M : List Nat := [0, 2, 0, 1]  -- Representing 1020 in base 3

theorem succeeding_number_in_base_3 :
  decimal_to_base_3 (base_3_to_decimal M + 1) = [1, 2, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_succeeding_number_in_base_3_l2669_266927


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l2669_266923

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l2669_266923


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2669_266924

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^8 + 3 * x^5 - 5) + 6 * (x^6 - 5 * x^3 + 4)

theorem sum_of_coefficients_is_zero : 
  polynomial 1 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2669_266924


namespace NUMINAMATH_CALUDE_f_less_than_g_l2669_266937

/-- Represents a board arrangement -/
def Board (m n : ℕ+) := Fin m → Fin n → Bool

/-- Number of arrangements with at least one row or column of noughts -/
def f (m n : ℕ+) : ℕ := sorry

/-- Number of arrangements with at least one row of noughts or column of crosses -/
def g (m n : ℕ+) : ℕ := sorry

/-- The theorem stating that f(m,n) < g(m,n) for all positive m and n -/
theorem f_less_than_g (m n : ℕ+) : f m n < g m n := by sorry

end NUMINAMATH_CALUDE_f_less_than_g_l2669_266937


namespace NUMINAMATH_CALUDE_cos_555_degrees_l2669_266925

theorem cos_555_degrees : Real.cos (555 * Real.pi / 180) = -(Real.sqrt 6 / 4 + Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_555_degrees_l2669_266925


namespace NUMINAMATH_CALUDE_melted_sphere_radius_l2669_266968

theorem melted_sphere_radius (r : ℝ) : 
  r > 0 → (4 / 3 * Real.pi * r^3 = 8 * (4 / 3 * Real.pi * 1^3)) → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_melted_sphere_radius_l2669_266968


namespace NUMINAMATH_CALUDE_set_representability_l2669_266903

-- Define the items
def item1 : Type := Unit  -- Placeholder for vague concept
def item2 : Set ℝ := {x : ℝ | x^2 + 3 = 0}
def item3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

-- Define a predicate for set representability
def is_set_representable (α : Type) : Prop := Nonempty (Set α)

-- State the theorem
theorem set_representability :
  ¬ is_set_representable item1 ∧ 
  is_set_representable item2 ∧ 
  is_set_representable item3 :=
sorry

end NUMINAMATH_CALUDE_set_representability_l2669_266903


namespace NUMINAMATH_CALUDE_laurent_series_expansion_l2669_266985

/-- The Laurent series expansion of f(z) = 1 / (z^2 - 1)^2 in the region 0 < |z-1| < 2 -/
theorem laurent_series_expansion (z : ℂ) (h : 0 < Complex.abs (z - 1) ∧ Complex.abs (z - 1) < 2) :
  (fun z => 1 / (z^2 - 1)^2) z = ∑' n, ((-1)^n * (n + 3) / 2^(n + 4)) * (z - 1)^n :=
sorry

end NUMINAMATH_CALUDE_laurent_series_expansion_l2669_266985


namespace NUMINAMATH_CALUDE_function_is_even_l2669_266943

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_is_even
  (f : ℝ → ℝ)
  (h1 : has_period f 4)
  (h2 : ∀ x, f (2 + x) = f (2 - x)) :
  is_even_function f :=
sorry

end NUMINAMATH_CALUDE_function_is_even_l2669_266943


namespace NUMINAMATH_CALUDE_vendor_first_day_sale_percentage_l2669_266945

/-- Represents the percentage of apples sold on the first day -/
def first_day_sale_percentage : ℝ := sorry

/-- Represents the total number of apples initially -/
def total_apples : ℝ := sorry

/-- Represents the number of apples remaining after the first day's sale -/
def apples_after_first_sale : ℝ := total_apples * (1 - first_day_sale_percentage)

/-- Represents the number of apples thrown away on the first day -/
def apples_thrown_first_day : ℝ := 0.2 * apples_after_first_sale

/-- Represents the number of apples remaining after throwing away on the first day -/
def apples_remaining_first_day : ℝ := apples_after_first_sale - apples_thrown_first_day

/-- Represents the number of apples sold on the second day -/
def apples_sold_second_day : ℝ := 0.5 * apples_remaining_first_day

/-- Represents the number of apples thrown away on the second day -/
def apples_thrown_second_day : ℝ := apples_remaining_first_day - apples_sold_second_day

/-- Represents the total number of apples thrown away -/
def total_apples_thrown : ℝ := apples_thrown_first_day + apples_thrown_second_day

theorem vendor_first_day_sale_percentage :
  first_day_sale_percentage = 0.5 ∧
  total_apples_thrown = 0.3 * total_apples :=
sorry

end NUMINAMATH_CALUDE_vendor_first_day_sale_percentage_l2669_266945


namespace NUMINAMATH_CALUDE_binomial_multiplication_l2669_266947

theorem binomial_multiplication (x : ℝ) : (4 * x - 3) * (x + 7) = 4 * x^2 + 25 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_multiplication_l2669_266947


namespace NUMINAMATH_CALUDE_f_inequality_range_l2669_266961

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x else Real.log x / Real.log 0.3

theorem f_inequality_range (t : ℝ) :
  (∀ x, f x ≤ t^2/4 - t + 1) ↔ t ∈ Set.Iic 1 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_range_l2669_266961


namespace NUMINAMATH_CALUDE_windows_per_floor_l2669_266938

theorem windows_per_floor (floors : ℕ) (payment_per_window : ℚ) 
  (deduction_per_3days : ℚ) (days_taken : ℕ) (final_payment : ℚ) :
  floors = 3 →
  payment_per_window = 2 →
  deduction_per_3days = 1 →
  days_taken = 6 →
  final_payment = 16 →
  ∃ (windows_per_floor : ℕ), 
    windows_per_floor = 3 ∧
    (floors * windows_per_floor * payment_per_window - 
      (days_taken / 3 : ℚ) * deduction_per_3days = final_payment) :=
by
  sorry

end NUMINAMATH_CALUDE_windows_per_floor_l2669_266938


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2669_266972

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2669_266972


namespace NUMINAMATH_CALUDE_geometric_locus_l2669_266967

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y^2 - x^2 = 0
def condition2 (x y : ℝ) : Prop := x^2 + y^2 = 4*(y - 1)
def condition3 (x : ℝ) : Prop := x^2 - 2*x + 1 = 0
def condition4 (x y : ℝ) : Prop := x^2 - 2*x*y + y^2 = -1

-- Define the theorem
theorem geometric_locus :
  (∀ x y : ℝ, condition1 x y ↔ (y = x ∨ y = -x)) ∧
  (∀ x y : ℝ, condition2 x y ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x : ℝ, condition3 x ↔ x = 1) ∧
  (¬∃ x y : ℝ, condition4 x y) :=
by sorry

end NUMINAMATH_CALUDE_geometric_locus_l2669_266967


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2669_266956

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x| + |x - 1| < a)) → a ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2669_266956


namespace NUMINAMATH_CALUDE_question_1_l2669_266994

theorem question_1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) :
  a^2 + 3/2 * b - 5 = -2 := by sorry

end NUMINAMATH_CALUDE_question_1_l2669_266994


namespace NUMINAMATH_CALUDE_vector_subtraction_l2669_266940

/-- Given vectors BA and CA in ℝ², prove that BC = BA - CA -/
theorem vector_subtraction (BA CA : ℝ × ℝ) (h1 : BA = (2, 3)) (h2 : CA = (4, 7)) :
  (BA.1 - CA.1, BA.2 - CA.2) = (-2, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2669_266940


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2669_266993

theorem square_difference_given_sum_and_product (m n : ℝ) 
  (h1 : m + n = 6) (h2 : m * n = 4) : (m - n)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2669_266993


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2669_266933

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 4 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 4 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2669_266933


namespace NUMINAMATH_CALUDE_cosine_equation_roots_l2669_266930

theorem cosine_equation_roots (θ : Real) :
  (0 ≤ θ) ∧ (θ < 360) →
  (3 * Real.cos θ + 1 / Real.cos θ = 4) →
  ∃ p : Nat, p = 3 := by sorry

end NUMINAMATH_CALUDE_cosine_equation_roots_l2669_266930


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l2669_266979

/-- Proves that for a square plot with an area of 289 sq ft and a total fencing cost of Rs. 3876, the price per foot of fencing is Rs. 57. -/
theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) (h1 : area = 289) (h2 : total_cost = 3876) :
  (total_cost / (4 * Real.sqrt area)) = 57 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l2669_266979


namespace NUMINAMATH_CALUDE_library_book_loan_l2669_266946

theorem library_book_loan (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) : 
  initial_books = 75 → 
  return_rate = 4/5 → 
  final_books = 64 → 
  (initial_books : ℚ) - final_books = (1 - return_rate) * 55 := by
  sorry

end NUMINAMATH_CALUDE_library_book_loan_l2669_266946


namespace NUMINAMATH_CALUDE_train_speed_problem_l2669_266977

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 210)
  (h2 : length2 = 260)
  (h3 : speed1 = 40)
  (h4 : time = 16.918646508279338)
  (h5 : length1 > 0)
  (h6 : length2 > 0)
  (h7 : speed1 > 0)
  (h8 : time > 0) :
  ∃ speed2 : ℝ, 
    speed2 > 0 ∧ 
    (length1 + length2) / 1000 = (speed1 + speed2) * (time / 3600) ∧
    speed2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2669_266977


namespace NUMINAMATH_CALUDE_expenditure_ratio_l2669_266912

/-- Given two persons P1 and P2 with the following conditions:
    - The ratio of their incomes is 5:4
    - Each saves Rs. 1800
    - The income of P1 is Rs. 4500
    Prove that the ratio of their expenditures is 3:2 -/
theorem expenditure_ratio (income_p1 income_p2 expenditure_p1 expenditure_p2 savings : ℕ) :
  income_p1 = 4500 ∧
  5 * income_p2 = 4 * income_p1 ∧
  savings = 1800 ∧
  income_p1 - expenditure_p1 = savings ∧
  income_p2 - expenditure_p2 = savings →
  3 * expenditure_p2 = 2 * expenditure_p1 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l2669_266912


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2669_266921

theorem min_value_expression (y : ℝ) :
  y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) ≥ 1/27 :=
sorry

theorem equality_condition :
  ∃ y : ℝ, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) = 1/27 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2669_266921


namespace NUMINAMATH_CALUDE_triangle_side_values_l2669_266976

/-- Given a triangle ABC with area S, sides b and c, prove that the third side a
    has one of two specific values. -/
theorem triangle_side_values (S b c : ℝ) (h1 : S = 12 * Real.sqrt 3)
    (h2 : b * c = 48) (h3 : b - c = 2) :
    ∃ (a : ℝ), (a = 2 * Real.sqrt 13 ∨ a = 2 * Real.sqrt 37) ∧
               (S = 1/2 * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_values_l2669_266976


namespace NUMINAMATH_CALUDE_train_passengers_l2669_266971

/-- The number of people on a train after three stops -/
def people_after_three_stops (initial : ℕ) 
  (stop1_off stop1_on : ℕ) 
  (stop2_off stop2_on : ℕ) 
  (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off + stop1_on - stop2_off + stop2_on - stop3_off + stop3_on

/-- Theorem stating the number of people on the train after three stops -/
theorem train_passengers : 
  people_after_three_stops 48 12 7 15 9 6 11 = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l2669_266971


namespace NUMINAMATH_CALUDE_range_of_m_l2669_266986

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q,
    prove that the range of values for m is m > 2/3. -/
theorem range_of_m (p q : ℝ → Prop) (m : ℝ) 
  (h_p : ∀ x, p x ↔ (x + 1) * (x - 1) ≤ 0)
  (h_q : ∀ x, q x ↔ (x + 1) * (x - (3 * m - 1)) ≤ 0)
  (h_m_pos : m > 0)
  (h_sufficient : ∀ x, p x → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬p x) :
  m > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2669_266986


namespace NUMINAMATH_CALUDE_number_exists_l2669_266988

theorem number_exists : ∃ x : ℝ, 0.6667 * x - 10 = 0.25 * x := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l2669_266988


namespace NUMINAMATH_CALUDE_quadratic_m_range_l2669_266995

def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  2 * x^2 - m * x + 1 = 0

def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂

def roots_in_range (m : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_equation m x → x ≥ 1/2 ∧ x ≤ 4

theorem quadratic_m_range :
  ∀ m : ℝ, (has_two_distinct_roots m ∧ roots_in_range m) ↔ (m > 2 * Real.sqrt 2 ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_m_range_l2669_266995


namespace NUMINAMATH_CALUDE_min_value_expression_l2669_266944

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a > b) (hbc : b > c) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2669_266944


namespace NUMINAMATH_CALUDE_english_only_enrollment_l2669_266909

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 45)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : german ≥ both) :
  total - german + both = 23 := by
  sorry

end NUMINAMATH_CALUDE_english_only_enrollment_l2669_266909


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l2669_266982

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_two : 2 * lg 5 + lg 4 = 2 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l2669_266982


namespace NUMINAMATH_CALUDE_triangle_theorem_l2669_266955

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3 ∧
  t.a + t.c = 6 ∧
  (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2) = (1/2) * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.B = π/3 ∧ (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2669_266955


namespace NUMINAMATH_CALUDE_turnip_zhuchka_weight_ratio_l2669_266911

/-- The weight ratio between Zhuchka and a cat -/
def zhuchka_cat_ratio : ℚ := 3

/-- The weight ratio between a cat and a mouse -/
def cat_mouse_ratio : ℚ := 10

/-- The weight ratio between a turnip and a mouse -/
def turnip_mouse_ratio : ℚ := 60

/-- The weight ratio between a turnip and Zhuchka -/
def turnip_zhuchka_ratio : ℚ := 2

theorem turnip_zhuchka_weight_ratio :
  turnip_mouse_ratio / (cat_mouse_ratio * zhuchka_cat_ratio) = turnip_zhuchka_ratio :=
by sorry

end NUMINAMATH_CALUDE_turnip_zhuchka_weight_ratio_l2669_266911


namespace NUMINAMATH_CALUDE_notebook_pages_calculation_l2669_266980

theorem notebook_pages_calculation (num_notebooks : ℕ) (pages_per_day : ℕ) (days_lasted : ℕ) : 
  num_notebooks > 0 → 
  pages_per_day > 0 → 
  days_lasted > 0 → 
  (pages_per_day * days_lasted) % num_notebooks = 0 → 
  (pages_per_day * days_lasted) / num_notebooks = 40 :=
by
  sorry

#check notebook_pages_calculation 5 4 50

end NUMINAMATH_CALUDE_notebook_pages_calculation_l2669_266980


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2669_266999

theorem complex_magnitude_problem (t : ℝ) (h : t > 0) :
  Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 5 → t = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2669_266999


namespace NUMINAMATH_CALUDE_gravitational_force_on_moon_l2669_266950

/-- Represents the gravitational force at a given distance from Earth's center -/
def gravitational_force (distance : ℝ) : ℝ := sorry

/-- The distance from Earth's center to its surface in miles -/
def earth_surface_distance : ℝ := 4000

/-- The distance from Earth's center to the moon in miles -/
def moon_distance : ℝ := 240000

/-- The gravitational force on Earth's surface in Newtons -/
def earth_surface_force : ℝ := 600

theorem gravitational_force_on_moon :
  gravitational_force earth_surface_distance = earth_surface_force →
  (∀ d : ℝ, gravitational_force d * d^2 = gravitational_force earth_surface_distance * earth_surface_distance^2) →
  gravitational_force moon_distance = 1/6 := by sorry

end NUMINAMATH_CALUDE_gravitational_force_on_moon_l2669_266950


namespace NUMINAMATH_CALUDE_tech_personnel_stats_l2669_266992

def intermediate_count : ℕ := 40
def senior_count : ℕ := 10
def total_count : ℕ := intermediate_count + senior_count

def intermediate_avg : ℝ := 35
def senior_avg : ℝ := 45

def intermediate_var : ℝ := 18
def senior_var : ℝ := 73

def total_avg : ℝ := 37
def total_var : ℝ := 45

theorem tech_personnel_stats :
  (intermediate_count * intermediate_avg + senior_count * senior_avg) / total_count = total_avg ∧
  ((intermediate_count * (intermediate_var + intermediate_avg^2) + 
    senior_count * (senior_var + senior_avg^2)) / total_count - total_avg^2) = total_var :=
by sorry

end NUMINAMATH_CALUDE_tech_personnel_stats_l2669_266992


namespace NUMINAMATH_CALUDE_abs_squared_minus_two_abs_minus_fifteen_solution_set_l2669_266998

theorem abs_squared_minus_two_abs_minus_fifteen_solution_set :
  {x : ℝ | |x|^2 - 2*|x| - 15 > 0} = {x : ℝ | x < -5 ∨ x > 5} := by
  sorry

end NUMINAMATH_CALUDE_abs_squared_minus_two_abs_minus_fifteen_solution_set_l2669_266998


namespace NUMINAMATH_CALUDE_land_profit_calculation_l2669_266948

/-- Represents the profit calculation for land distribution among sons -/
theorem land_profit_calculation (total_land : ℝ) (num_sons : ℕ) 
  (profit_per_unit : ℝ) (unit_area : ℝ) (hectare_to_sqm : ℝ) : 
  total_land = 3 ∧ 
  num_sons = 8 ∧ 
  profit_per_unit = 500 ∧ 
  unit_area = 750 ∧ 
  hectare_to_sqm = 10000 → 
  (total_land * hectare_to_sqm / num_sons / unit_area * profit_per_unit * 4 : ℝ) = 10000 := by
  sorry

#check land_profit_calculation

end NUMINAMATH_CALUDE_land_profit_calculation_l2669_266948


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l2669_266926

theorem x_range_for_inequality (x : ℝ) : 
  (0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3) ↔ 
  (∀ y : ℝ, y > 0 → (2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y)) / (x + y) > 3 * x^2 * y) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l2669_266926


namespace NUMINAMATH_CALUDE_fish_left_in_tank_l2669_266934

def fish_tank_problem (initial_fish : ℕ) (fish_taken_out : ℕ) : Prop :=
  initial_fish ≥ fish_taken_out ∧ 
  initial_fish - fish_taken_out = 3

theorem fish_left_in_tank : fish_tank_problem 19 16 := by
  sorry

end NUMINAMATH_CALUDE_fish_left_in_tank_l2669_266934


namespace NUMINAMATH_CALUDE_circle_equation_radius_l2669_266959

theorem circle_equation_radius (x y d : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 2*y + d = 0) → 
  (∃ h k : ℝ, ∀ x y, (x - h)^2 + (y - k)^2 = 5^2) →
  d = -8 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l2669_266959


namespace NUMINAMATH_CALUDE_floor_of_e_l2669_266920

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem floor_of_e : ⌊e⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l2669_266920


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l2669_266919

theorem other_root_of_complex_quadratic (z : ℂ) :
  z = 4 + 7*I ∧ z^2 = -73 + 24*I → (-z)^2 = -73 + 24*I := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l2669_266919


namespace NUMINAMATH_CALUDE_hyperbola_min_focal_length_l2669_266939

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    focal length 2c, and a + b - c = 2, the minimum value of 2c is 4 + 4√2. -/
theorem hyperbola_min_focal_length (a b c : ℝ) : 
  a > 0 → b > 0 → a + b - c = 2 → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  2 * c ≥ 4 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_min_focal_length_l2669_266939


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l2669_266970

theorem sin_product_equals_one_sixteenth : 
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * 
  Real.sin (54 * π / 180) * Real.sin (72 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l2669_266970


namespace NUMINAMATH_CALUDE_inequality_solution_l2669_266996

theorem inequality_solution : 
  let x : ℝ := 3
  (1/3 - x/3 : ℝ) < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2669_266996


namespace NUMINAMATH_CALUDE_profit_ratio_of_partners_l2669_266914

theorem profit_ratio_of_partners (p q : ℕ) (investment_ratio : Rat) (time_p time_q : ℕ) 
  (h1 : investment_ratio = 7 / 5)
  (h2 : time_p = 7)
  (h3 : time_q = 14) :
  (p : Rat) / q = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_profit_ratio_of_partners_l2669_266914


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2669_266935

/-- An arithmetic sequence is a sequence where the difference between 
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem stating that for an arithmetic sequence satisfying 
    the given condition, 2a_9 - a_10 = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2669_266935


namespace NUMINAMATH_CALUDE_sand_bag_fraction_l2669_266990

/-- Given a bag of sand weighing 50 kg, prove that after using 30 kg,
    the remaining sand accounts for 2/5 of the total bag. -/
theorem sand_bag_fraction (total_weight : ℝ) (used_weight : ℝ) 
  (h1 : total_weight = 50)
  (h2 : used_weight = 30) :
  (total_weight - used_weight) / total_weight = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sand_bag_fraction_l2669_266990


namespace NUMINAMATH_CALUDE_molar_mass_X1_l2669_266904

-- Define the substances
def X1 : String := "CuO"
def X2 : String := "Cu"
def X3 : String := "CuSO4"
def X4 : String := "Cu(OH)2"

-- Define the molar masses
def molar_mass_Cu : Float := 63.5
def molar_mass_O : Float := 16.0

-- Define the chemical reactions
def reaction1 : String := "X1 + H2 → X2 + H2O"
def reaction2 : String := "X2 + H2SO4 → X3 + H2"
def reaction3 : String := "X3 + 2KOH → X4 + K2SO4"
def reaction4 : String := "X4 → X1 + H2O"

-- Define the properties of the substances
def X1_properties : String := "black powder"
def X2_properties : String := "red-colored substance"
def X3_properties : String := "blue-colored solution"
def X4_properties : String := "blue precipitate"

-- Theorem to prove
theorem molar_mass_X1 : 
  molar_mass_Cu + molar_mass_O = 79.5 := by sorry

end NUMINAMATH_CALUDE_molar_mass_X1_l2669_266904


namespace NUMINAMATH_CALUDE_min_value_product_l2669_266908

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 9) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 57 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a + b/a + c/b + a/c = 9 ∧
    (a/b + b/c + c/a) * (b/a + c/b + a/c) = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2669_266908


namespace NUMINAMATH_CALUDE_parallel_lines_a_perpendicular_lines_a_l2669_266910

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - 3/2 = 0

-- Parallel lines condition
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Perpendicular lines condition
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- Theorem for parallel lines
theorem parallel_lines_a (a : ℝ) :
  parallel a → a = 4 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines_a (a : ℝ) :
  perpendicular a → a = 0 ∨ a = -20/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_perpendicular_lines_a_l2669_266910


namespace NUMINAMATH_CALUDE_no_solution_l2669_266983

-- Define the system of equations
def system (x₁ x₂ x₃ : ℝ) : Prop :=
  (x₁ + 4*x₂ + 10*x₃ = 1) ∧
  (0*x₁ - 5*x₂ - 13*x₃ = -1.25) ∧
  (0*x₁ + 0*x₂ + 0*x₃ = 1.25)

-- Theorem stating that the system has no solution
theorem no_solution : ¬∃ (x₁ x₂ x₃ : ℝ), system x₁ x₂ x₃ := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2669_266983


namespace NUMINAMATH_CALUDE_membership_condition_l2669_266963

def is_necessary_but_not_sufficient {α : Type*} (A B : Set α) : Prop :=
  (A ∩ B = B) ∧ (A ≠ B) ∧
  (∀ x, x ∈ B → x ∈ A) ∧
  (∃ x, x ∈ A ∧ x ∉ B)

theorem membership_condition {α : Type*} (A B : Set α) 
  (h1 : A ∩ B = B) (h2 : A ≠ B) :
  is_necessary_but_not_sufficient A B :=
sorry

end NUMINAMATH_CALUDE_membership_condition_l2669_266963


namespace NUMINAMATH_CALUDE_habitable_earth_surface_fraction_l2669_266966

theorem habitable_earth_surface_fraction :
  let total_surface : ℚ := 1
  let water_covered_fraction : ℚ := 2/3
  let land_fraction : ℚ := 1 - water_covered_fraction
  let inhabitable_land_fraction : ℚ := 2/3
  inhabitable_land_fraction * land_fraction = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_habitable_earth_surface_fraction_l2669_266966


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l2669_266991

theorem imaginary_part_of_complex_division (z : ℂ) : 
  z = (3 + 4 * I) / I → Complex.im z = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l2669_266991


namespace NUMINAMATH_CALUDE_area_of_triangle_fyh_l2669_266969

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  ef : ℝ
  gh : ℝ
  area : ℝ

/-- Theorem: Area of triangle FYH in a trapezoid with specific measurements -/
theorem area_of_triangle_fyh (t : Trapezoid) 
  (h1 : t.ef = 24)
  (h2 : t.gh = 36)
  (h3 : t.area = 360) :
  let height : ℝ := 2 * t.area / (t.ef + t.gh)
  let area_egh : ℝ := (1 / 2) * t.gh * height
  let area_efh : ℝ := t.area - area_egh
  let height_eyh : ℝ := (2 / 5) * height
  let area_efh_recalc : ℝ := (1 / 2) * t.ef * (height - height_eyh)
  area_efh - area_efh_recalc = 86.4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_fyh_l2669_266969


namespace NUMINAMATH_CALUDE_distinct_roots_find_m_l2669_266929

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m^2 + m

-- Define the discriminant
def discriminant (m : ℝ) : ℝ := (-(2*m + 1))^2 - 4*(m^2 + m)

-- Define the condition for the roots
def root_condition (a b : ℝ) : Prop := (2*a + b) * (a + 2*b) = 20

-- Theorem 1: The equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : discriminant m > 0 :=
sorry

-- Theorem 2: When the root condition is satisfied, m = -2 or m = 1
theorem find_m (m : ℝ) :
  (∃ a b : ℝ, quadratic m a = 0 ∧ quadratic m b = 0 ∧ a ≠ b ∧ root_condition a b) →
  m = -2 ∨ m = 1 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_find_m_l2669_266929


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_factorials_l2669_266987

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 5 + factorial 6 + factorial 7

theorem largest_prime_factor_of_sum_of_factorials :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_factorials ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_factorials → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_factorials_l2669_266987


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l2669_266906

/-- A real polynomial of degree 3 -/
structure Polynomial3 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of the polynomial at a point x -/
def Polynomial3.eval (p : Polynomial3) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The condition that |p(x)| ≤ 1 for all x such that |x| ≤ 1 -/
def BoundedOnUnitInterval (p : Polynomial3) : Prop :=
  ∀ x : ℝ, |x| ≤ 1 → |p.eval x| ≤ 1

/-- The theorem statement -/
theorem polynomial_coefficient_bound (p : Polynomial3) 
  (h : BoundedOnUnitInterval p) : 
  |p.a| + |p.b| + |p.c| + |p.d| ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l2669_266906
