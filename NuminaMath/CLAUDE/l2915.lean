import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_condition_l2915_291588

theorem divisibility_condition (a b : ℕ+) (h : b ≥ 2) :
  (2^a.val + 1) % (2^b.val - 1) = 0 ↔ b = 2 ∧ a.val % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2915_291588


namespace NUMINAMATH_CALUDE_solution_set_equality_l2915_291556

-- Define the solution set of |8x+9| < 7
def solution_set : Set ℝ := {x : ℝ | |8*x + 9| < 7}

-- Define the inequality ax^2 + bx > 2
def inequality (a b : ℝ) (x : ℝ) : Prop := a*x^2 + b*x > 2

-- State the theorem
theorem solution_set_equality (a b : ℝ) : 
  (∀ x : ℝ, x ∈ solution_set ↔ inequality a b x) → a + b = -13 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2915_291556


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l2915_291577

-- Define a decreasing function f
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define the function f passing through specific points
def f_passes_through (f : ℝ → ℝ) : Prop :=
  f 0 = 3 ∧ f 3 = -1

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | |f (x + 1) - 1| < 2}

theorem solution_set_is_open_interval
  (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (h_passes_through : f_passes_through f) :
  solution_set f = Set.Ioo (-1 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l2915_291577


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2915_291564

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 222 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 222 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2915_291564


namespace NUMINAMATH_CALUDE_johns_money_to_father_l2915_291524

def initial_amount : ℚ := 200
def fraction_to_mother : ℚ := 3/8
def amount_left : ℚ := 65

theorem johns_money_to_father :
  (initial_amount - fraction_to_mother * initial_amount - amount_left) / initial_amount = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_to_father_l2915_291524


namespace NUMINAMATH_CALUDE_rabbit_population_estimate_l2915_291584

/-- Capture-recapture estimation of rabbit population -/
theorem rabbit_population_estimate :
  ∀ (total_population : ℕ)
    (first_capture second_capture recaptured_tagged : ℕ),
  first_capture = 10 →
  second_capture = 10 →
  recaptured_tagged = 2 →
  total_population = (first_capture * second_capture) / recaptured_tagged →
  total_population = 50 :=
by
  sorry

#check rabbit_population_estimate

end NUMINAMATH_CALUDE_rabbit_population_estimate_l2915_291584


namespace NUMINAMATH_CALUDE_exists_good_board_bad_after_recolor_l2915_291543

/-- Represents a 6x6 board with two colors -/
def Board := Fin 6 → Fin 6 → Bool

/-- Checks if a cell has a neighboring cell of the same color -/
def has_same_color_neighbor (b : Board) (i j : Fin 6) : Prop :=
  (i > 0 ∧ b (i-1) j = b i j) ∨
  (i < 5 ∧ b (i+1) j = b i j) ∨
  (j > 0 ∧ b i (j-1) = b i j) ∨
  (j < 5 ∧ b i (j+1) = b i j)

/-- Checks if the board is good (each cell has a same-color neighbor) -/
def is_good (b : Board) : Prop :=
  ∀ i j, has_same_color_neighbor b i j

/-- Recolors a row of the board -/
def recolor_row (b : Board) (row : Fin 6) : Board :=
  λ i j => if i = row then ¬(b i j) else b i j

/-- Recolors a column of the board -/
def recolor_column (b : Board) (col : Fin 6) : Board :=
  λ i j => if j = col then ¬(b i j) else b i j

/-- The main theorem: there exists a good board that becomes bad after any row or column recoloring -/
theorem exists_good_board_bad_after_recolor :
  ∃ b : Board, is_good b ∧
    (∀ row, ¬(is_good (recolor_row b row))) ∧
    (∀ col, ¬(is_good (recolor_column b col))) :=
sorry

end NUMINAMATH_CALUDE_exists_good_board_bad_after_recolor_l2915_291543


namespace NUMINAMATH_CALUDE_square_root_of_64_l2915_291506

theorem square_root_of_64 : ∃ x : ℝ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
  sorry

end NUMINAMATH_CALUDE_square_root_of_64_l2915_291506


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l2915_291507

theorem smallest_integer_in_consecutive_set : 
  ∀ (n : ℤ), 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) → 
  n ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l2915_291507


namespace NUMINAMATH_CALUDE_greatest_prime_producing_integer_l2915_291544

def f (x : ℤ) : ℤ := |5 * x^2 - 52 * x + 21|

def is_greatest_prime_producing_integer (n : ℤ) : Prop :=
  Nat.Prime (f n).natAbs ∧
  ∀ m : ℤ, m > n → ¬(Nat.Prime (f m).natAbs)

theorem greatest_prime_producing_integer :
  is_greatest_prime_producing_integer 10 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_producing_integer_l2915_291544


namespace NUMINAMATH_CALUDE_inverse_function_point_l2915_291515

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x / Real.log a

def has_inverse_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g x = y

theorem inverse_function_point (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  has_inverse_point (f a) 2 4 →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_function_point_l2915_291515


namespace NUMINAMATH_CALUDE_class_division_theorem_l2915_291568

theorem class_division_theorem :
  ∀ (x : ℕ),
  x ≤ 26 ∧ x ≤ 30 →
  x - (24 - (30 - x)) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_class_division_theorem_l2915_291568


namespace NUMINAMATH_CALUDE_nail_polish_difference_l2915_291516

theorem nail_polish_difference (kim heidi karen : ℕ) : 
  kim = 12 →
  heidi = kim + 5 →
  karen + heidi = 25 →
  karen < kim →
  kim - karen = 4 := by
sorry

end NUMINAMATH_CALUDE_nail_polish_difference_l2915_291516


namespace NUMINAMATH_CALUDE_total_factories_to_check_l2915_291520

theorem total_factories_to_check (first_group : ℕ) (second_group : ℕ) (remaining : ℕ) :
  first_group = 69 → second_group = 52 → remaining = 48 →
  first_group + second_group + remaining = 169 := by
  sorry

end NUMINAMATH_CALUDE_total_factories_to_check_l2915_291520


namespace NUMINAMATH_CALUDE_second_number_form_l2915_291538

theorem second_number_form (G S : ℕ) (h1 : G = 4) 
  (h2 : ∃ k : ℕ, 1642 = k * G + 6) 
  (h3 : ∃ l : ℕ, S = l * G + 4) : 
  ∃ m : ℕ, S = 4 * m + 4 := by
sorry

end NUMINAMATH_CALUDE_second_number_form_l2915_291538


namespace NUMINAMATH_CALUDE_negation_equivalence_l2915_291554

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2915_291554


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l2915_291529

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of possible arrangements for a given configuration -/
def arrangements (a b c : ℕ) : ℕ := sorry

/-- The probability of stacking crates to a specific height -/
def stack_probability (crate_dims : CrateDimensions) (num_crates target_height : ℕ) : ℚ :=
  sorry

theorem crate_stacking_probability :
  let crate_dims : CrateDimensions := ⟨2, 5, 7⟩
  let num_crates : ℕ := 8
  let target_height : ℕ := 36
  stack_probability crate_dims num_crates target_height = 98 / 6561 := by sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l2915_291529


namespace NUMINAMATH_CALUDE_river_flow_volume_l2915_291537

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 12) 
  (h_width : width = 35) 
  (h_flow_rate : flow_rate_kmph = 8) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 56000 := by
  sorry

end NUMINAMATH_CALUDE_river_flow_volume_l2915_291537


namespace NUMINAMATH_CALUDE_volume_of_region_l2915_291501

def region (x y z : ℝ) : Prop :=
  abs (x + y + z) + abs (x + y - z) ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem volume_of_region : 
  MeasureTheory.volume {p : ℝ × ℝ × ℝ | region p.1 p.2.1 p.2.2} = 108 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l2915_291501


namespace NUMINAMATH_CALUDE_base_conversion_2546_to_base5_l2915_291569

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c d : Nat) : Nat :=
  a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

/-- Theorem stating that 2546 (base 10) is equal to 4141 (base 5) --/
theorem base_conversion_2546_to_base5 :
  base5ToBase10 4 1 4 1 = 2546 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_2546_to_base5_l2915_291569


namespace NUMINAMATH_CALUDE_unique_modulus_of_quadratic_roots_l2915_291598

theorem unique_modulus_of_quadratic_roots :
  ∃! r : ℝ, ∃ z : ℂ, z^2 - 6*z + 34 = 0 ∧ Complex.abs z = r :=
by sorry

end NUMINAMATH_CALUDE_unique_modulus_of_quadratic_roots_l2915_291598


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2915_291574

/-- The radius of a circle inscribed in a sector that is one-third of a circle --/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  let r := (5 * Real.sqrt 3 - 5) / 2
  r > 0 ∧ r + r * Real.sqrt 3 = R :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2915_291574


namespace NUMINAMATH_CALUDE_amanda_notebooks_l2915_291534

theorem amanda_notebooks (initial : ℕ) : 
  initial + 6 - 2 = 14 → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l2915_291534


namespace NUMINAMATH_CALUDE_smallest_clock_equivalent_is_nine_l2915_291525

/-- A number is clock equivalent to its square if it's congruent to its square modulo 12 -/
def IsClockEquivalent (n : ℕ) : Prop := n ≡ n^2 [MOD 12]

/-- The smallest number greater than 5 that is clock equivalent to its square -/
def SmallestClockEquivalent : ℕ := 9

theorem smallest_clock_equivalent_is_nine :
  IsClockEquivalent SmallestClockEquivalent ∧
  ∀ n : ℕ, 5 < n ∧ n < SmallestClockEquivalent → ¬IsClockEquivalent n :=
by sorry

end NUMINAMATH_CALUDE_smallest_clock_equivalent_is_nine_l2915_291525


namespace NUMINAMATH_CALUDE_inequality_of_means_l2915_291559

theorem inequality_of_means (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : (x - y)^2 > 396*x*y) (h2 : 2.0804*x*y > x^2 + y^2) :
  1.01 * Real.sqrt (x*y) > (x + y)/2 ∧ (x + y)/2 > 100 * (2*x*y/(x + y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_means_l2915_291559


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2915_291522

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + count → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, consecutive_nonprimes n 6 ∧ 
           is_prime (n + 6) ∧ 
           ∀ m : ℕ, m < n → ¬(consecutive_nonprimes m 6 ∧ is_prime (m + 6)) ∧
           n + 6 = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2915_291522


namespace NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_1_l2915_291521

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_1 :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_1_l2915_291521


namespace NUMINAMATH_CALUDE_both_make_shots_probability_l2915_291596

/-- The probability that both person A and person B make their shots -/
def prob_both_make_shots (prob_A prob_B : ℝ) : ℝ := prob_A * prob_B

theorem both_make_shots_probability :
  let prob_A : ℝ := 2/5
  let prob_B : ℝ := 1/2
  prob_both_make_shots prob_A prob_B = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_both_make_shots_probability_l2915_291596


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2915_291570

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (-1, 2) and b = (2, m), then m = -4. -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  b = (2, m) →
  (∃ (k : ℝ), b = k • a) →
  m = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2915_291570


namespace NUMINAMATH_CALUDE_symmetry_of_f_2x_l2915_291523

def center_of_symmetry (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ k : ℤ, p = (k * Real.pi / 2 - Real.pi / 8, 0)}

theorem symmetry_of_f_2x (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (-x) = 3 * Real.cos x - Real.sin x) :
  center_of_symmetry (fun x ↦ f (2 * x)) = 
    {p : ℝ × ℝ | ∃ k : ℤ, p = (k * Real.pi / 2 - Real.pi / 8, 0)} := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_f_2x_l2915_291523


namespace NUMINAMATH_CALUDE_power_multiplication_l2915_291519

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2915_291519


namespace NUMINAMATH_CALUDE_probability_of_drawing_parts_l2915_291510

def total_parts : ℕ := 10
def drawn_parts : ℕ := 6

def prob_draw_one (n m k : ℕ) : ℚ :=
  (n.choose k) / (m.choose k)

def prob_draw_two (n m k : ℕ) : ℚ :=
  ((n-2).choose (k-2)) / (m.choose k)

theorem probability_of_drawing_parts :
  (prob_draw_one (total_parts - 1) total_parts drawn_parts = 3/5) ∧
  (prob_draw_two (total_parts - 2) total_parts drawn_parts = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_parts_l2915_291510


namespace NUMINAMATH_CALUDE_gold_distribution_l2915_291540

theorem gold_distribution (n : ℕ) (a₁ : ℚ) (d : ℚ) : 
  n = 10 → 
  (4 * a₁ + 6 * d = 3) → 
  (3 * a₁ + 24 * d = 4) → 
  d = 7/78 :=
by sorry

end NUMINAMATH_CALUDE_gold_distribution_l2915_291540


namespace NUMINAMATH_CALUDE_sum_of_digits_c_equals_five_l2915_291562

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def a : ℕ := sum_of_digits (4568^777)
def b : ℕ := sum_of_digits a
def c : ℕ := sum_of_digits b

theorem sum_of_digits_c_equals_five : c = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_c_equals_five_l2915_291562


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2915_291509

/-- Represents the state of tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure ExchangeBooth where
  input_color : String
  input_amount : ℕ
  output_silver : ℕ
  output_other_color : String
  output_other_amount : ℕ

/-- Performs a single exchange at a booth if possible -/
def exchange (state : TokenState) (booth : ExchangeBooth) : Option TokenState :=
  sorry

/-- Performs all possible exchanges until no more are possible -/
def exchange_all (initial_state : TokenState) (booths : List ExchangeBooth) : TokenState :=
  sorry

/-- The main theorem to prove -/
theorem max_silver_tokens : 
  let initial_state : TokenState := ⟨100, 65, 0⟩
  let booths : List ExchangeBooth := [
    ⟨"red", 3, 1, "blue", 2⟩,
    ⟨"blue", 4, 1, "red", 2⟩
  ]
  let final_state := exchange_all initial_state booths
  final_state.silver = 65 :=
sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2915_291509


namespace NUMINAMATH_CALUDE_amanda_coffee_blend_typeA_quantity_l2915_291567

/-- Represents the cost and quantity of coffee in Amanda's Coffee Shop blend --/
structure CoffeeBlend where
  typeA_cost : ℝ
  typeB_cost : ℝ
  typeA_quantity : ℝ
  typeB_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the quantity of type A coffee in the blend --/
theorem amanda_coffee_blend_typeA_quantity (blend : CoffeeBlend) 
  (h1 : blend.typeA_cost = 4.60)
  (h2 : blend.typeB_cost = 5.95)
  (h3 : blend.typeB_quantity = 2 * blend.typeA_quantity)
  (h4 : blend.total_cost = 511.50)
  (h5 : blend.total_cost = blend.typeA_cost * blend.typeA_quantity + blend.typeB_cost * blend.typeB_quantity) :
  blend.typeA_quantity = 31 := by
  sorry


end NUMINAMATH_CALUDE_amanda_coffee_blend_typeA_quantity_l2915_291567


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2915_291503

def U : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Finset ℕ := {0, 1, 3, 5, 8}
def B : Finset ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2915_291503


namespace NUMINAMATH_CALUDE_triangle_height_l2915_291579

theorem triangle_height (C b area h : Real) : 
  C = π / 3 → 
  b = 4 → 
  area = 2 * Real.sqrt 3 → 
  area = (1 / 2) * b * h * Real.sin C → 
  h = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l2915_291579


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_3150_l2915_291505

theorem sum_of_prime_factors_3150 : (Finset.sum (Finset.filter Nat.Prime (Finset.range (3150 + 1))) id) = 17 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_3150_l2915_291505


namespace NUMINAMATH_CALUDE_five_mile_taxi_cost_l2915_291514

/-- Calculates the cost of a taxi ride given the base fare, cost per mile, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_fare + cost_per_mile * distance

/-- Proves that a 5-mile taxi ride costs $2.75 given the specified base fare and cost per mile. -/
theorem five_mile_taxi_cost :
  let base_fare : ℝ := 1.50
  let cost_per_mile : ℝ := 0.25
  let distance : ℝ := 5
  taxi_cost base_fare cost_per_mile distance = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_five_mile_taxi_cost_l2915_291514


namespace NUMINAMATH_CALUDE_time_to_destination_l2915_291589

/-- The time it takes to reach a destination given relative speeds and distances -/
theorem time_to_destination
  (your_speed : ℝ)
  (harris_speed : ℝ)
  (harris_time : ℝ)
  (distance_ratio : ℝ)
  (h1 : your_speed = 3 * harris_speed)
  (h2 : harris_time = 3)
  (h3 : distance_ratio = 5) :
  your_speed * (distance_ratio * harris_time / your_speed) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_time_to_destination_l2915_291589


namespace NUMINAMATH_CALUDE_good_carrots_count_l2915_291550

theorem good_carrots_count (vanessa_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : vanessa_carrots = 17)
  (h2 : mom_carrots = 14)
  (h3 : bad_carrots = 7) :
  vanessa_carrots + mom_carrots - bad_carrots = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l2915_291550


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_range_l2915_291555

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(1-a)*x + 2

-- State the theorem
theorem decreasing_function_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Iic 4, ∀ y ∈ Set.Iic 4, x < y → f a x > f a y) →
  a ∈ Set.Ici 5 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_range_l2915_291555


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2915_291586

theorem quadratic_equation_unique_solution :
  ∃! x : ℝ, x^2 + 2*x + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2915_291586


namespace NUMINAMATH_CALUDE_suit_price_calculation_suit_price_theorem_l2915_291539

theorem suit_price_calculation (original_price : ℝ) 
  (increase_rate : ℝ) (reduction_rate : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - reduction_rate)
  final_price

theorem suit_price_theorem : 
  suit_price_calculation 300 0.2 0.1 = 324 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_suit_price_theorem_l2915_291539


namespace NUMINAMATH_CALUDE_problem_solution_l2915_291592

noncomputable section

variable (g : ℝ → ℝ)

-- g is invertible
variable (h : Function.Bijective g)

-- Define the values of g given in the table
axiom g_1 : g 1 = 4
axiom g_2 : g 2 = 6
axiom g_3 : g 3 = 9
axiom g_4 : g 4 = 10
axiom g_5 : g 5 = 12

-- The theorem to prove
theorem problem_solution :
  g (g 2) + g (Function.invFun g 12) + Function.invFun g (Function.invFun g 10) = 25 := by
  sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2915_291592


namespace NUMINAMATH_CALUDE_least_coins_l2915_291561

theorem least_coins (n : ℕ) : 
  (n > 0) → 
  (n % 7 = 3) → 
  (n % 5 = 4) → 
  (∀ m : ℕ, m > 0 → m % 7 = 3 → m % 5 = 4 → n ≤ m) → 
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_least_coins_l2915_291561


namespace NUMINAMATH_CALUDE_green_face_probability_l2915_291563

/-- The probability of rolling a green face on a 10-sided die with 3 green faces is 3/10. -/
theorem green_face_probability (total_faces : ℕ) (green_faces : ℕ) 
  (h1 : total_faces = 10) (h2 : green_faces = 3) : 
  (green_faces : ℚ) / total_faces = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_l2915_291563


namespace NUMINAMATH_CALUDE_qizhi_median_is_65_l2915_291528

/-- Represents the homework duration data for a group of students -/
structure HomeworkData where
  durations : List Nat
  counts : List Nat
  total_students : Nat

/-- Calculates the median of a dataset given its HomeworkData -/
def calculate_median (data : HomeworkData) : Rat :=
  sorry

/-- The specific homework data for the problem -/
def qizhi_data : HomeworkData :=
  { durations := [50, 60, 70, 80],
    counts := [14, 11, 10, 15],
    total_students := 50 }

/-- Theorem stating that the median of the given homework data is 65 minutes -/
theorem qizhi_median_is_65 : calculate_median qizhi_data = 65 := by
  sorry

end NUMINAMATH_CALUDE_qizhi_median_is_65_l2915_291528


namespace NUMINAMATH_CALUDE_cubic_inequality_l2915_291533

theorem cubic_inequality (x : ℝ) : 
  x^3 - 12*x^2 + 36*x + 8 > 0 ↔ x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2915_291533


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l2915_291526

/-- Calculates the number of hours a mechanic worked given the total cost, 
    cost of parts, and labor rate per minute. -/
theorem mechanic_work_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (num_parts : ℕ) 
  (labor_rate_per_minute : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_cost = 20) 
  (h3 : num_parts = 2) 
  (h4 : labor_rate_per_minute = 0.5) : 
  (total_cost - part_cost * num_parts) / (labor_rate_per_minute * 60) = 6 := by
sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l2915_291526


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2915_291599

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → b = 10 → 
  (a + b + c) / 3 = a + 5 → 
  (a + b + c) / 3 = c - 20 → 
  a + b + c = 75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2915_291599


namespace NUMINAMATH_CALUDE_angle_EFG_is_60_degrees_l2915_291571

-- Define the angles as real numbers
variable (x : ℝ)
variable (angle_CFG angle_CEB angle_BEA angle_EFG : ℝ)

-- Define the parallel lines property
variable (AD_parallel_FG : Prop)

-- State the theorem
theorem angle_EFG_is_60_degrees 
  (h1 : AD_parallel_FG)
  (h2 : angle_CFG = 1.5 * x)
  (h3 : angle_CEB = x)
  (h4 : angle_BEA = 2 * x)
  (h5 : angle_EFG = angle_CFG) :
  angle_EFG = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_EFG_is_60_degrees_l2915_291571


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l2915_291566

def total_people : ℕ := 18
def num_tribes : ℕ := 3
def people_per_tribe : ℕ := 6
def num_quitters : ℕ := 3

theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_people num_quitters
  let same_tribe_ways := num_tribes * Nat.choose people_per_tribe num_quitters
  (same_tribe_ways : ℚ) / total_ways = 5 / 68 := by
    sorry

end NUMINAMATH_CALUDE_survivor_quitters_probability_l2915_291566


namespace NUMINAMATH_CALUDE_unique_base7_digit_l2915_291512

/-- Converts a base 7 number of the form 52x4₇ to base 10 --/
def base7ToBase10 (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

/-- The set of valid digits in base 7 --/
def base7Digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6}

theorem unique_base7_digit : 
  ∃! x : ℕ, x ∈ base7Digits ∧ isDivisibleBy19 (base7ToBase10 x) := by sorry

end NUMINAMATH_CALUDE_unique_base7_digit_l2915_291512


namespace NUMINAMATH_CALUDE_not_both_follow_control_principle_option_d_is_incorrect_l2915_291551

/-- Represents an experimental approach -/
inductive ExperimentalApproach
| BlankControl
| RepeatWithSameSoil

/-- Represents a scientific principle -/
inductive ScientificPrinciple
| Control
| Repeatability

/-- Function to determine which principle an approach follows -/
def principleFollowed (approach : ExperimentalApproach) : ScientificPrinciple :=
  match approach with
  | ExperimentalApproach.BlankControl => ScientificPrinciple.Control
  | ExperimentalApproach.RepeatWithSameSoil => ScientificPrinciple.Repeatability

/-- Theorem stating that not both approaches follow the control principle -/
theorem not_both_follow_control_principle :
  ¬(principleFollowed ExperimentalApproach.BlankControl = ScientificPrinciple.Control ∧
     principleFollowed ExperimentalApproach.RepeatWithSameSoil = ScientificPrinciple.Control) :=
by sorry

/-- Main theorem proving that the statement in option D is incorrect -/
theorem option_d_is_incorrect :
  ¬(∀ (approach : ExperimentalApproach), principleFollowed approach = ScientificPrinciple.Control) :=
by sorry

end NUMINAMATH_CALUDE_not_both_follow_control_principle_option_d_is_incorrect_l2915_291551


namespace NUMINAMATH_CALUDE_year_2078_is_wu_xu_l2915_291502

/-- Represents the Heavenly Stems in the Chinese calendar system -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Chinese calendar system -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Chinese calendar system -/
structure ChineseYear where
  stem : HeavenlyStem
  branch : EarthlyBranch

/-- The number of Heavenly Stems -/
def numHeavenlyStems : Nat := 10

/-- The number of Earthly Branches -/
def numEarthlyBranches : Nat := 12

/-- The starting year of the reform and opening up period -/
def reformStartYear : Nat := 1978

/-- Function to get the next Heavenly Stem in the cycle -/
def nextHeavenlyStem (s : HeavenlyStem) : HeavenlyStem := sorry

/-- Function to get the next Earthly Branch in the cycle -/
def nextEarthlyBranch (b : EarthlyBranch) : EarthlyBranch := sorry

/-- Function to get the Chinese Year representation for a given year -/
def getChineseYear (year : Nat) : ChineseYear := sorry

/-- Theorem stating that the year 2078 is represented as "Wu Xu" -/
theorem year_2078_is_wu_xu :
  let year2016 := ChineseYear.mk HeavenlyStem.Bing EarthlyBranch.Shen
  let year2078 := getChineseYear 2078
  year2078 = ChineseYear.mk HeavenlyStem.Wu EarthlyBranch.Xu := by
  sorry

end NUMINAMATH_CALUDE_year_2078_is_wu_xu_l2915_291502


namespace NUMINAMATH_CALUDE_frog_paths_count_l2915_291593

/-- Represents a triangular grid -/
structure TriangularGrid :=
  (top_row_squares : ℕ)
  (total_squares : ℕ)

/-- Represents the possible moves of the frog -/
inductive Move
  | down
  | down_left

/-- Calculates the number of distinct paths in a triangular grid -/
def count_distinct_paths (grid : TriangularGrid) : ℕ :=
  sorry

/-- Theorem stating the number of distinct paths in the specific grid -/
theorem frog_paths_count (grid : TriangularGrid) 
  (h1 : grid.top_row_squares = 5)
  (h2 : grid.total_squares = 29) :
  count_distinct_paths grid = 256 :=
sorry

end NUMINAMATH_CALUDE_frog_paths_count_l2915_291593


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2915_291581

theorem quadratic_inequality (z : ℝ) : z^2 - 40*z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2915_291581


namespace NUMINAMATH_CALUDE_f_sum_theorem_l2915_291580

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_theorem : f π + (deriv f) (π / 2) = -3 / π := by
  sorry

end NUMINAMATH_CALUDE_f_sum_theorem_l2915_291580


namespace NUMINAMATH_CALUDE_custom_op_two_five_l2915_291583

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

/-- Theorem stating that 2 ⊗ 5 = 23 under the custom operation -/
theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_five_l2915_291583


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2915_291576

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 6 ∧ ∃ (x' y' : ℝ), 4 < x' ∧ x' < 8 ∧ 8 < y' ∧ y' < 12 ∧ (⌊y'⌋ - ⌈x'⌉ : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2915_291576


namespace NUMINAMATH_CALUDE_polynomial_integer_roots_l2915_291557

theorem polynomial_integer_roots (p : ℤ → ℤ) 
  (h1 : ∃ a : ℤ, p a = 1) 
  (h3 : ∃ b : ℤ, p b = 3) : 
  ¬(∃ y1 y2 : ℤ, y1 ≠ y2 ∧ p y1 = 2 ∧ p y2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_integer_roots_l2915_291557


namespace NUMINAMATH_CALUDE_interest_equality_l2915_291536

theorem interest_equality (P : ℝ) : 
  let I₁ := P * 0.04 * 5
  let I₂ := P * 0.05 * 4
  I₁ = I₂ ∧ I₁ = 20 := by sorry

end NUMINAMATH_CALUDE_interest_equality_l2915_291536


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2915_291572

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x - 2) * (x + 2) < 5} = {x : ℝ | -3 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2915_291572


namespace NUMINAMATH_CALUDE_min_distance_to_equidistant_point_l2915_291547

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Point P is equidistant from C₁ and C₂ -/
def equidistant (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 9 = x^2 + y^2 + 2*x + 2*y + 1

/-- The minimum distance from the origin to any point equidistant from C₁ and C₂ is 4/5 -/
theorem min_distance_to_equidistant_point :
  ∃ (x₀ y₀ : ℝ), equidistant x₀ y₀ ∧
    ∀ (x y : ℝ), equidistant x y → x₀^2 + y₀^2 ≤ x^2 + y^2 ∧
    x₀^2 + y₀^2 = (4/5)^2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_equidistant_point_l2915_291547


namespace NUMINAMATH_CALUDE_fifth_day_distance_l2915_291578

def running_distance (day : ℕ) : ℕ :=
  2 + (day - 1)

theorem fifth_day_distance : running_distance 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fifth_day_distance_l2915_291578


namespace NUMINAMATH_CALUDE_problem_statement_l2915_291587

theorem problem_statement (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (4/x + 1/y ≥ 4/m + 1/n)) ∧
  (4/m + 1/n ≥ 9) ∧
  (Real.sqrt m + Real.sqrt n ≤ Real.sqrt 2) ∧
  (m > n → 1/(m-1) < 1/(n-1)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2915_291587


namespace NUMINAMATH_CALUDE_equation_positive_root_l2915_291548

theorem equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l2915_291548


namespace NUMINAMATH_CALUDE_selection_theorem_l2915_291549

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of representatives to be selected -/
def num_representatives : ℕ := 3

theorem selection_theorem :
  (choose total_people num_representatives = 35) ∧
  (choose num_girls 1 * choose num_boys 2 +
   choose num_girls 2 * choose num_boys 1 +
   choose num_girls 3 = 31) ∧
  (choose total_people num_representatives -
   choose num_boys num_representatives -
   choose num_girls num_representatives = 30) := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2915_291549


namespace NUMINAMATH_CALUDE_math_test_score_distribution_l2915_291541

theorem math_test_score_distribution (total_students : ℕ) (percentile_80_score : ℕ) :
  total_students = 1200 →
  percentile_80_score = 103 →
  (∃ (students_above_threshold : ℕ),
    students_above_threshold ≥ 240 ∧
    students_above_threshold = total_students - (total_students * 80 / 100)) := by
  sorry

end NUMINAMATH_CALUDE_math_test_score_distribution_l2915_291541


namespace NUMINAMATH_CALUDE_quadratic_inequality_rational_inequality_l2915_291530

-- Problem 1
theorem quadratic_inequality (x : ℝ) :
  2 * x^2 - 3 * x + 1 < 0 ↔ 1/2 < x ∧ x < 1 :=
by sorry

-- Problem 2
theorem rational_inequality (x : ℝ) :
  2 * x / (x + 1) ≥ 1 ↔ x ≥ 1 ∨ x < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_rational_inequality_l2915_291530


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l2915_291553

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- The axes of symmetry are the coordinate axes -/
  axes_are_coordinate_axes : True
  /-- One endpoint of minor axis and two foci form equilateral triangle -/
  equilateral_triangle : b / c = Real.sqrt 3
  /-- Foci are on the y-axis -/
  foci_on_y_axis : True
  /-- Relation between a and c -/
  a_minus_c : a - c = Real.sqrt 3
  /-- Pythagorean theorem for ellipse -/
  ellipse_relation : a^2 = b^2 + c^2

/-- The equation of the special ellipse -/
def ellipse_equation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, y^2 / 12 + x^2 / 9 = 1 ↔ y^2 / e.a^2 + x^2 / e.b^2 = 1

/-- The main theorem about the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) : ellipse_equation e := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l2915_291553


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l2915_291582

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the cross-section area of a sliced cylinder -/
def crossSectionArea (c : Cylinder) (arcAngle : ℝ) : ℝ :=
  sorry

theorem cylinder_cross_section_area :
  let c : Cylinder := { radius := 8, height := 5 }
  let arcAngle : ℝ := 90 * π / 180  -- 90 degrees in radians
  crossSectionArea c arcAngle = 16 * π * Real.sqrt 2 + 32 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l2915_291582


namespace NUMINAMATH_CALUDE_max_intersections_l2915_291504

/-- Given 15 points on the positive x-axis and 10 points on the positive y-axis,
    with segments connecting each point on the x-axis to each point on the y-axis,
    the maximum number of intersection points in the interior of the first quadrant is 4725. -/
theorem max_intersections (x_points y_points : ℕ) (h1 : x_points = 15) (h2 : y_points = 10) :
  (x_points.choose 2) * (y_points.choose 2) = 4725 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_l2915_291504


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2915_291500

theorem constant_term_binomial_expansion (n : ℕ+) :
  (∃ k : ℕ, k ≤ n ∧ 3*n = 4*k) → n ≠ 6 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2915_291500


namespace NUMINAMATH_CALUDE_seating_arrangement_l2915_291558

theorem seating_arrangement (total_people : ℕ) (rows_of_nine : ℕ) (rows_of_ten : ℕ) : 
  total_people = 54 →
  total_people = 9 * rows_of_nine + 10 * rows_of_ten →
  rows_of_nine > 0 →
  rows_of_ten = 0 := by
sorry

end NUMINAMATH_CALUDE_seating_arrangement_l2915_291558


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2915_291527

/-- A rectangular block made of 1-cm cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of cubes in the block -/
def Block.volume (b : Block) : ℕ := b.length * b.width * b.height

/-- The number of cubes not visible when three faces are shown -/
def Block.hiddenCubes (b : Block) : ℕ := (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- One dimension is at least 5 -/
def Block.hasLargeDimension (b : Block) : Prop :=
  b.length ≥ 5 ∨ b.width ≥ 5 ∨ b.height ≥ 5

theorem smallest_block_volume (b : Block) :
  b.hiddenCubes = 252 →
  b.hasLargeDimension →
  ∀ b' : Block, b'.hiddenCubes = 252 → b'.hasLargeDimension → b.volume ≤ b'.volume →
  b.volume = 280 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2915_291527


namespace NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l2915_291532

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  2^n - 2

/-- Theorem stating that f(n) correctly represents the sum of numbers in the nth row -/
theorem sum_of_nth_row (n : ℕ) : 
  f n = 2^n - 2 :=
sorry

/-- Corollary for the 100th row -/
theorem sum_of_100th_row : 
  f 100 = 2^100 - 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l2915_291532


namespace NUMINAMATH_CALUDE_complex_number_problem_l2915_291542

theorem complex_number_problem (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10 * Complex.I ∧ 
  z₂ = 3 - 4 * Complex.I ∧ 
  1 / z = 1 / z₁ + 1 / z₂ → 
  z = 5 - (5 / 2) * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2915_291542


namespace NUMINAMATH_CALUDE_simple_interest_proof_l2915_291508

/-- Given a principal amount for which the compound interest at 5% per annum for 2 years is 56.375,
    prove that the simple interest at 5% per annum for 2 years is 55. -/
theorem simple_interest_proof (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 56.375 → P * 0.05 * 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_proof_l2915_291508


namespace NUMINAMATH_CALUDE_outfits_count_l2915_291560

/-- The number of outfits that can be made with the given conditions -/
def num_outfits : ℕ :=
  let red_shirts := 6
  let green_shirts := 7
  let pants := 9
  let blue_hats := 10
  let red_hats := 10
  let red_shirt_outfits := red_shirts * pants * blue_hats
  let green_shirt_outfits := green_shirts * pants * red_hats
  red_shirt_outfits + green_shirt_outfits

/-- Theorem stating that the number of outfits is 1170 -/
theorem outfits_count : num_outfits = 1170 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2915_291560


namespace NUMINAMATH_CALUDE_ordering_proof_l2915_291513

theorem ordering_proof (x a y : ℝ) (h1 : x < a) (h2 : a < y) (h3 : y < 0) :
  x^3 < y*a^2 ∧ y*a^2 < a*y ∧ a*y < x^2 := by
  sorry

end NUMINAMATH_CALUDE_ordering_proof_l2915_291513


namespace NUMINAMATH_CALUDE_find_b_value_l2915_291552

theorem find_b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2915_291552


namespace NUMINAMATH_CALUDE_adjacent_points_probability_l2915_291517

/-- The number of points around the square -/
def n : ℕ := 12

/-- The number of pairs of adjacent points -/
def adjacent_pairs : ℕ := 12

/-- The total number of ways to choose 2 points from n points -/
def total_combinations : ℕ := n * (n - 1) / 2

/-- The probability of choosing two adjacent points -/
def probability : ℚ := adjacent_pairs / total_combinations

theorem adjacent_points_probability : probability = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_points_probability_l2915_291517


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2915_291535

-- Define set A
def A : Set ℝ := {y | ∃ x, y = |x|}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 1 - 2*x - x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2915_291535


namespace NUMINAMATH_CALUDE_second_car_traveled_5km_l2915_291590

/-- Represents the distance traveled by the second car -/
def second_car_distance : ℝ := 5

/-- The initial distance between the two cars -/
def initial_distance : ℝ := 105

/-- The distance traveled by the first car before turning back -/
def first_car_distance : ℝ := 25 + 15 + 25

/-- The final distance between the two cars -/
def final_distance : ℝ := 20

/-- Theorem stating that the second car traveled 5 km -/
theorem second_car_traveled_5km :
  initial_distance - (first_car_distance + 15 + second_car_distance) = final_distance :=
by sorry

end NUMINAMATH_CALUDE_second_car_traveled_5km_l2915_291590


namespace NUMINAMATH_CALUDE_roots_sum_condition_l2915_291565

theorem roots_sum_condition (a b : ℤ) (α : ℝ) :
  (0 ≤ α ∧ α < 2 * Real.pi) →
  (∀ x : ℝ, x^2 + a * x + 2 * b^2 = 0 ↔ x = Real.sin α ∨ x = Real.cos α) →
  a + b = 1 ∨ a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_condition_l2915_291565


namespace NUMINAMATH_CALUDE_log_equation_solution_l2915_291585

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y ^ 2 / Real.log 3 + Real.log y / Real.log (1/3) = 6 →
  y = 729 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2915_291585


namespace NUMINAMATH_CALUDE_train_crossing_time_l2915_291595

/-- Time taken for a train to cross a man running in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 450 →
  train_speed = 60 * 1000 / 3600 →
  man_speed = 6 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2915_291595


namespace NUMINAMATH_CALUDE_tournament_games_count_l2915_291591

/-- A single-elimination tournament structure -/
structure Tournament :=
  (total_teams : ℕ)
  (bye_teams : ℕ)
  (h_bye : bye_teams ≤ total_teams)

/-- The number of games played in a single-elimination tournament -/
def games_played (t : Tournament) : ℕ :=
  t.total_teams - 1

theorem tournament_games_count (t : Tournament) 
  (h_total : t.total_teams = 32) 
  (h_bye : t.bye_teams = 8) : 
  games_played t = 32 := by
sorry

end NUMINAMATH_CALUDE_tournament_games_count_l2915_291591


namespace NUMINAMATH_CALUDE_points_collinear_l2915_291597

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def is_acute_angled (t : Triangle) : Prop := sorry

-- Define the angle A to be 60°
def angle_A_is_60_degrees (t : Triangle) : Prop := sorry

-- Define the orthocenter H
def orthocenter (t : Triangle) : Point := sorry

-- Define point M
def point_M (t : Triangle) (H : Point) : Point := sorry

-- Define point N
def point_N (t : Triangle) (H : Point) : Point := sorry

-- Define the circumcenter O
def circumcenter (t : Triangle) : Point := sorry

-- Define collinearity
def collinear (P Q R S : Point) : Prop := sorry

-- Theorem statement
theorem points_collinear (t : Triangle) (H : Point) (M N O : Point) :
  is_acute_angled t →
  angle_A_is_60_degrees t →
  H = orthocenter t →
  M = point_M t H →
  N = point_N t H →
  O = circumcenter t →
  collinear M N H O :=
sorry

end NUMINAMATH_CALUDE_points_collinear_l2915_291597


namespace NUMINAMATH_CALUDE_equation_solution_l2915_291518

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 ↔ x = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2915_291518


namespace NUMINAMATH_CALUDE_equal_share_money_l2915_291546

theorem equal_share_money (emani_money : ℕ) (difference : ℕ) : 
  emani_money = 150 →
  difference = 30 →
  (emani_money + (emani_money - difference)) / 2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_money_l2915_291546


namespace NUMINAMATH_CALUDE_matrix_inverse_and_solution_l2915_291545

theorem matrix_inverse_and_solution (A B M : Matrix (Fin 2) (Fin 2) ℝ) : 
  A = ![![2, 0], ![-1, 1]] →
  B = ![![2, 4], ![3, 5]] →
  A * M = B →
  A⁻¹ = ![![1/2, 0], ![1/2, 1]] ∧
  M = ![![1, 2], ![4, 7]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_and_solution_l2915_291545


namespace NUMINAMATH_CALUDE_absolute_value_of_c_l2915_291575

theorem absolute_value_of_c (a b c : ℤ) : 
  a * (3 + I : ℂ)^4 + b * (3 + I : ℂ)^3 + c * (3 + I : ℂ)^2 + b * (3 + I : ℂ) + a = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 116 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_of_c_l2915_291575


namespace NUMINAMATH_CALUDE_expected_voters_for_candidate_A_l2915_291531

theorem expected_voters_for_candidate_A (total_voters : ℝ) (dem_percent : ℝ) 
  (dem_for_A : ℝ) (rep_for_A : ℝ) (h1 : dem_percent = 0.6) 
  (h2 : dem_for_A = 0.75) (h3 : rep_for_A = 0.3) : 
  (dem_percent * dem_for_A + (1 - dem_percent) * rep_for_A) * 100 = 57 := by
  sorry

end NUMINAMATH_CALUDE_expected_voters_for_candidate_A_l2915_291531


namespace NUMINAMATH_CALUDE_production_theorem_l2915_291573

/-- Represents the production process with recycling --/
def max_parts_and_waste (initial_blanks : ℕ) (efficiency : ℚ) : ℕ × ℚ :=
  sorry

/-- The theorem statement --/
theorem production_theorem :
  max_parts_and_waste 20 (2/3) = (29, 1/3) := by sorry

end NUMINAMATH_CALUDE_production_theorem_l2915_291573


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2915_291511

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def B : Set ℝ := {x : ℝ | x ≥ 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2915_291511


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2915_291594

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  2 * x₁^2 - 3 * x₁ + 1 = 0 ∧ 2 * x₂^2 - 3 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2915_291594
