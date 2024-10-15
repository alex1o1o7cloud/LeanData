import Mathlib

namespace NUMINAMATH_CALUDE_spades_in_deck_l1887_188762

/-- 
Given a deck of 52 cards containing some spades, prove that if the probability 
of not drawing a spade on the first draw is 0.75, then there are 13 spades in the deck.
-/
theorem spades_in_deck (total_cards : ℕ) (prob_not_spade : ℚ) (num_spades : ℕ) : 
  total_cards = 52 →
  prob_not_spade = 3/4 →
  (total_cards - num_spades : ℚ) / total_cards = prob_not_spade →
  num_spades = 13 := by
  sorry

end NUMINAMATH_CALUDE_spades_in_deck_l1887_188762


namespace NUMINAMATH_CALUDE_phi_values_l1887_188778

theorem phi_values (φ : Real) : 
  Real.sqrt 3 * Real.sin (20 * π / 180) = 2 * Real.cos φ - Real.sin φ → 
  φ = 140 * π / 180 ∨ φ = 40 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_phi_values_l1887_188778


namespace NUMINAMATH_CALUDE_congruent_side_length_l1887_188758

-- Define the triangle
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ
  side : ℝ

-- Define our specific triangle
def ourTriangle : IsoscelesTriangle where
  base := 24
  area := 60
  side := 13

-- Theorem statement
theorem congruent_side_length (t : IsoscelesTriangle) 
  (h1 : t.base = 24) 
  (h2 : t.area = 60) : 
  t.side = 13 := by
  sorry

#check congruent_side_length

end NUMINAMATH_CALUDE_congruent_side_length_l1887_188758


namespace NUMINAMATH_CALUDE_max_multiplication_table_sum_l1887_188794

theorem max_multiplication_table_sum : 
  ∀ (a b c d e f : ℕ), 
    a ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    b ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    c ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    d ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    e ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    f ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → 
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → 
    c ≠ d → c ≠ e → c ≠ f → 
    d ≠ e → d ≠ f → 
    e ≠ f → 
    (a * d + a * e + a * f + b * d + b * e + b * f + c * d + c * e + c * f) ≤ 961 :=
by sorry

end NUMINAMATH_CALUDE_max_multiplication_table_sum_l1887_188794


namespace NUMINAMATH_CALUDE_equation_solution_l1887_188759

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1887_188759


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l1887_188716

theorem equivalence_of_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/a + 1/b = Real.sqrt (a*b)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l1887_188716


namespace NUMINAMATH_CALUDE_rubys_math_homework_l1887_188774

/-- Ruby's math homework problem -/
theorem rubys_math_homework :
  ∀ (ruby_math ruby_reading nina_math nina_reading : ℕ),
  ruby_reading = 2 →
  nina_math = 4 * ruby_math →
  nina_reading = 8 * ruby_reading →
  nina_math + nina_reading = 48 →
  ruby_math = 6 := by
sorry

end NUMINAMATH_CALUDE_rubys_math_homework_l1887_188774


namespace NUMINAMATH_CALUDE_linear_function_property_l1887_188708

/-- A linear function f where f(6) - f(2) = 12 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) ∧ 
  (f 6 - f 2 = 12)

theorem linear_function_property (f : ℝ → ℝ) (h : LinearFunction f) : 
  f 12 - f 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1887_188708


namespace NUMINAMATH_CALUDE_neighbor_rolls_l1887_188754

def total_rolls : ℕ := 12
def grandmother_rolls : ℕ := 3
def uncle_rolls : ℕ := 4
def rolls_left : ℕ := 2

theorem neighbor_rolls : 
  total_rolls - grandmother_rolls - uncle_rolls - rolls_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_rolls_l1887_188754


namespace NUMINAMATH_CALUDE_a_44_mod_45_l1887_188726

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that the remainder when a_44 is divided by 45 is 9 -/
theorem a_44_mod_45 : a 44 % 45 = 9 := by sorry

end NUMINAMATH_CALUDE_a_44_mod_45_l1887_188726


namespace NUMINAMATH_CALUDE_min_a_correct_l1887_188701

/-- The number of cards in the deck -/
def n : ℕ := 52

/-- The probability that Alex and Dylan are on the same team given Alex's card number a -/
def p (a : ℕ) : ℚ :=
  let lower := (n - (a + 6) + 1).choose 2
  let higher := (a - 1).choose 2
  (lower + higher : ℚ) / (n - 2).choose 2

/-- The minimum value of a such that p(a) ≥ 1/2 -/
def min_a : ℕ := 14

theorem min_a_correct :
  (∀ a < min_a, p a < 1/2) ∧ p min_a ≥ 1/2 :=
sorry

#eval min_a

end NUMINAMATH_CALUDE_min_a_correct_l1887_188701


namespace NUMINAMATH_CALUDE_geometric_progression_sum_l1887_188752

theorem geometric_progression_sum (p q : ℝ) : 
  p ≠ q →                  -- Two distinct geometric progressions
  p + q = 3 →              -- Sum of common ratios is 3
  1 * p^5 + 1 * q^5 = 573 →  -- Sum of sixth terms is 573 (1 is the first term)
  1 * p^4 + 1 * q^4 = 161    -- Sum of fifth terms is 161
  := by sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_l1887_188752


namespace NUMINAMATH_CALUDE_luis_task_completion_l1887_188745

-- Define the start time and end time of the third task
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes
def end_third_task : Nat := 12 * 60 + 30  -- 12:30 PM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem luis_task_completion :
  ∀ (task_duration : Nat),
  (end_third_task - start_time = 3 * task_duration) →
  (start_time + num_tasks * task_duration = 13 * 60 + 40) :=
by
  sorry


end NUMINAMATH_CALUDE_luis_task_completion_l1887_188745


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l1887_188732

theorem largest_triangle_perimeter :
  ∀ x : ℕ,
  (7 + 9 > x) ∧ (7 + x > 9) ∧ (9 + x > 7) →
  (∀ y : ℕ, (7 + 9 > y) ∧ (7 + y > 9) ∧ (9 + y > 7) → x ≥ y) →
  7 + 9 + x = 31 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l1887_188732


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1887_188796

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1887_188796


namespace NUMINAMATH_CALUDE_jason_blue_marbles_count_l1887_188799

/-- The number of blue marbles Jason and Tom have in total -/
def total_blue_marbles : ℕ := 68

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := total_blue_marbles - tom_blue_marbles

theorem jason_blue_marbles_count : jason_blue_marbles = 44 := by
  sorry

end NUMINAMATH_CALUDE_jason_blue_marbles_count_l1887_188799


namespace NUMINAMATH_CALUDE_infinite_sum_equals_floor_l1887_188764

noncomputable def infiniteSum (x : ℝ) : ℕ → ℝ
  | 0 => ⌊(x + 1) / 2⌋
  | n + 1 => infiniteSum x n + ⌊(x + 2^(n+1)) / 2^(n+2)⌋

theorem infinite_sum_equals_floor (x : ℝ) :
  (∀ y : ℝ, ⌊2 * y⌋ = ⌊y⌋ + ⌊y + 1/2⌋) →
  (∃ N : ℕ, ∀ n ≥ N, ⌊(x + 2^n) / 2^(n+1)⌋ = 0) →
  (∃ M : ℕ, ∀ m ≥ M, infiniteSum x m = ⌊x⌋) :=
by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_floor_l1887_188764


namespace NUMINAMATH_CALUDE_vector_perpendicular_m_l1887_188775

theorem vector_perpendicular_m (a b : ℝ × ℝ) (m : ℝ) : 
  a = (3, 4) → 
  b = (2, -1) → 
  (a.1 + m * b.1, a.2 + m * b.2) • (a.1 - b.1, a.2 - b.2) = 0 → 
  m = 23 / 3 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_m_l1887_188775


namespace NUMINAMATH_CALUDE_bell_pepper_pieces_l1887_188787

/-- The number of bell peppers Tamia has -/
def num_peppers : ℕ := 5

/-- The number of large slices each pepper is cut into -/
def slices_per_pepper : ℕ := 20

/-- The number of smaller pieces each selected large slice is cut into -/
def pieces_per_slice : ℕ := 3

/-- The total number of bell pepper pieces Tamia will have -/
def total_pieces : ℕ := 
  let total_slices := num_peppers * slices_per_pepper
  let slices_to_cut := total_slices / 2
  let smaller_pieces := slices_to_cut * pieces_per_slice
  let remaining_slices := total_slices - slices_to_cut
  smaller_pieces + remaining_slices

theorem bell_pepper_pieces : total_pieces = 200 := by
  sorry

end NUMINAMATH_CALUDE_bell_pepper_pieces_l1887_188787


namespace NUMINAMATH_CALUDE_stride_sync_l1887_188706

/-- The least common multiple of Jack and Jill's stride lengths -/
def stride_lcm (jack_stride jill_stride : ℕ) : ℕ :=
  Nat.lcm jack_stride jill_stride

/-- Theorem stating that the LCM of Jack and Jill's strides is 448 cm -/
theorem stride_sync (jack_stride jill_stride : ℕ) 
  (h1 : jack_stride = 64) 
  (h2 : jill_stride = 56) : 
  stride_lcm jack_stride jill_stride = 448 := by
  sorry

end NUMINAMATH_CALUDE_stride_sync_l1887_188706


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1887_188705

theorem min_value_and_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m : ℝ, m = 6 ∧
    (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 →
      1 / a'^3 + 1 / b'^3 + 1 / c'^3 + 3 * a' * b' * c' ≥ m) ∧
    (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 →
      1 / a'^3 + 1 / b'^3 + 1 / c'^3 + 3 * a' * b' * c' = m →
      a' = 1 ∧ b' = 1 ∧ c' = 1)) ∧
  (∀ x : ℝ, abs (x + 1) - 2 * x < 6 ↔ x > -7/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1887_188705


namespace NUMINAMATH_CALUDE_circus_ticket_sales_l1887_188756

theorem circus_ticket_sales (total_tickets : ℕ) (adult_price children_price : ℚ) 
  (total_receipts : ℚ) (h1 : total_tickets = 522) 
  (h2 : adult_price = 15) (h3 : children_price = 8) (h4 : total_receipts = 5086) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * children_price = total_receipts ∧
    adult_tickets = 130 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_sales_l1887_188756


namespace NUMINAMATH_CALUDE_fishing_line_sections_l1887_188740

/-- The number of reels of fishing line John buys -/
def num_reels : ℕ := 3

/-- The length of fishing line in each reel (in meters) -/
def reel_length : ℕ := 100

/-- The length of each section John cuts the fishing line into (in meters) -/
def section_length : ℕ := 10

/-- The total number of sections John gets from cutting all the fishing line -/
def total_sections : ℕ := (num_reels * reel_length) / section_length

theorem fishing_line_sections :
  total_sections = 30 := by sorry

end NUMINAMATH_CALUDE_fishing_line_sections_l1887_188740


namespace NUMINAMATH_CALUDE_triplet_sum_position_l1887_188784

theorem triplet_sum_position 
  (x : Fin 6 → ℝ) 
  (s : Fin 20 → ℝ) 
  (h_order : ∀ i j, i < j → x i < x j) 
  (h_sums : ∀ i j, i < j → s i < s j) 
  (h_distinct : ∀ i j k l m n, i < j → j < k → l < m → m < n → 
    x i + x j + x k ≠ x l + x m + x n) 
  (h_s11 : x 1 + x 2 + x 3 = s 10) 
  (h_s15 : x 1 + x 2 + x 5 = s 14) : 
  x 0 + x 1 + x 5 = s 6 := by
sorry

end NUMINAMATH_CALUDE_triplet_sum_position_l1887_188784


namespace NUMINAMATH_CALUDE_red_light_probability_l1887_188780

theorem red_light_probability (p_first : ℝ) (p_both : ℝ) :
  p_first = 1/2 →
  p_both = 1/5 →
  p_both / p_first = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_red_light_probability_l1887_188780


namespace NUMINAMATH_CALUDE_parabola_intersection_l1887_188795

/-- Two parabolas intersect at exactly two points -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x - 5
  let g (x : ℝ) := x^2 - 2 * x + 3
  ∃! (s : Set (ℝ × ℝ)), s = {(1, -14), (4, -5)} ∧ 
    ∀ (x y : ℝ), (x, y) ∈ s ↔ f x = g x ∧ y = f x := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1887_188795


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1887_188786

theorem x_one_minus_f_equals_one :
  let α : ℝ := 3 + 2 * Real.sqrt 2
  let x : ℝ := α ^ 50
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1887_188786


namespace NUMINAMATH_CALUDE_negation_of_universal_absolute_value_l1887_188763

theorem negation_of_universal_absolute_value :
  (¬ ∀ x : ℝ, x = |x|) ↔ (∃ x : ℝ, x ≠ |x|) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_absolute_value_l1887_188763


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l1887_188769

/-- A line with slope 3/4 passing through (-12, -39) intersects the x-axis at x = 40 -/
theorem line_intersection_x_axis :
  ∀ (f : ℝ → ℝ),
  (∀ x y, f y - f x = (3/4) * (y - x)) →  -- Slope condition
  f (-12) = -39 →                         -- Point condition
  ∃ x, f x = 0 ∧ x = 40 :=                -- Intersection with x-axis
by
  sorry


end NUMINAMATH_CALUDE_line_intersection_x_axis_l1887_188769


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1887_188739

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + Complex.I) / (1 - Complex.I)^2
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1887_188739


namespace NUMINAMATH_CALUDE_eric_containers_l1887_188717

/-- The number of containers Eric has for his colored pencils. -/
def number_of_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) : ℕ :=
  (initial_pencils + additional_pencils) / pencils_per_container

theorem eric_containers :
  number_of_containers 150 30 36 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eric_containers_l1887_188717


namespace NUMINAMATH_CALUDE_base5_division_theorem_l1887_188741

/-- Converts a base-5 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base-5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Represents a number in base-5 --/
structure Base5Number where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 5

/-- Division operation for Base5Number --/
def base5Div (a b : Base5Number) : Base5Number :=
  { digits := toBase5 ((toDecimal a.digits) / (toDecimal b.digits))
    valid := sorry }

theorem base5_division_theorem :
  let a : Base5Number := ⟨[1, 0, 3, 2], sorry⟩  -- 2301 in base 5
  let b : Base5Number := ⟨[2, 2], sorry⟩        -- 22 in base 5
  let result : Base5Number := ⟨[2, 0, 1], sorry⟩  -- 102 in base 5
  base5Div a b = result := by sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l1887_188741


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1887_188782

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The total number of possible triangles formed by choosing 3 vertices from n vertices -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles with exactly one side being a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon (i.e., formed by three consecutive vertices) -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1887_188782


namespace NUMINAMATH_CALUDE_x_0_interval_l1887_188777

theorem x_0_interval (x_0 : ℝ) (h1 : x_0 ∈ Set.Ioo 0 π) 
  (h2 : Real.sin x_0 + Real.cos x_0 = 2/3) : 
  x_0 ∈ Set.Ioo (7*π/12) (3*π/4) := by
  sorry

end NUMINAMATH_CALUDE_x_0_interval_l1887_188777


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l1887_188736

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 6

/-- The number of apples Adam has -/
def adams_apples : ℕ := jackies_apples + 3

theorem adam_has_more_apples : adams_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_more_apples_l1887_188736


namespace NUMINAMATH_CALUDE_parabola_triangle_circumradius_range_l1887_188703

/-- A point on a parabola y = x^2 -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- Triangle on a parabola y = x^2 -/
structure ParabolaTriangle where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The circumradius of a triangle -/
def circumradius (t : ParabolaTriangle) : ℝ :=
  sorry  -- Definition of circumradius

theorem parabola_triangle_circumradius_range :
  ∀ (t : ParabolaTriangle), circumradius t > 1/2 ∧
  ∀ (r : ℝ), r > 1/2 → ∃ (t : ParabolaTriangle), circumradius t = r :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_circumradius_range_l1887_188703


namespace NUMINAMATH_CALUDE_product_of_values_l1887_188753

theorem product_of_values (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 24 * Real.rpow 3 (1/3))
  (h2 : x * z = 40 * Real.rpow 3 (1/3))
  (h3 : y * z = 15 * Real.rpow 3 (1/3)) :
  x * y * z = 72 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_values_l1887_188753


namespace NUMINAMATH_CALUDE_product_of_roots_l1887_188709

theorem product_of_roots (k m x₁ x₂ : ℝ) (h_distinct : x₁ ≠ x₂)
  (h₁ : 4 * x₁^2 - k * x₁ - m = 0) (h₂ : 4 * x₂^2 - k * x₂ - m = 0) :
  x₁ * x₂ = -m / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1887_188709


namespace NUMINAMATH_CALUDE_total_potatoes_brought_home_l1887_188749

/-- The number of people who received potatoes -/
def num_people : ℕ := 3

/-- The number of potatoes each person received -/
def potatoes_per_person : ℕ := 8

/-- Theorem: The total number of potatoes brought home is 24 -/
theorem total_potatoes_brought_home : 
  num_people * potatoes_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_brought_home_l1887_188749


namespace NUMINAMATH_CALUDE_exists_respectful_quadratic_with_zero_at_neg_one_l1887_188788

/-- A respectful quadratic polynomial. -/
structure RespectfulQuadratic where
  a : ℝ
  b : ℝ

/-- The polynomial function for a respectful quadratic. -/
def q (p : RespectfulQuadratic) (x : ℝ) : ℝ :=
  x^2 + p.a * x + p.b

/-- The condition that q(q(x)) = 0 has exactly four real roots. -/
def hasFourRoots (p : RespectfulQuadratic) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    ∀ (x : ℝ), q p (q p x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄

/-- The main theorem stating the existence of a respectful quadratic polynomial
    satisfying the required conditions. -/
theorem exists_respectful_quadratic_with_zero_at_neg_one :
  ∃ (p : RespectfulQuadratic), hasFourRoots p ∧ q p (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_respectful_quadratic_with_zero_at_neg_one_l1887_188788


namespace NUMINAMATH_CALUDE_claudia_weekend_earnings_l1887_188712

-- Define the charge per class
def charge_per_class : ℝ := 10.00

-- Define the number of kids in Saturday's class
def saturday_attendance : ℕ := 20

-- Define the number of kids in Sunday's class
def sunday_attendance : ℕ := saturday_attendance / 2

-- Define the total attendance for the weekend
def total_attendance : ℕ := saturday_attendance + sunday_attendance

-- Theorem to prove
theorem claudia_weekend_earnings :
  (total_attendance : ℝ) * charge_per_class = 300.00 := by
  sorry

end NUMINAMATH_CALUDE_claudia_weekend_earnings_l1887_188712


namespace NUMINAMATH_CALUDE_letters_ratio_l1887_188742

/-- Proves that the ratio of letters Greta's mother received to the total letters Greta and her brother received is 2:1 -/
theorem letters_ratio (greta_letters brother_letters mother_letters : ℕ) : 
  greta_letters = brother_letters + 10 →
  brother_letters = 40 →
  greta_letters + brother_letters + mother_letters = 270 →
  ∃ k : ℕ, mother_letters = k * (greta_letters + brother_letters) →
  mother_letters = 2 * (greta_letters + brother_letters) := by
sorry

end NUMINAMATH_CALUDE_letters_ratio_l1887_188742


namespace NUMINAMATH_CALUDE_lunch_packing_days_l1887_188707

/-- Represents the number of school days for each school -/
structure SchoolDays where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of days a student packs lunch -/
structure LunchDays where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- Given the conditions of the problem, prove the correct expressions for lunch packing days -/
theorem lunch_packing_days (sd : SchoolDays) : 
  ∃ (ld : LunchDays), 
    ld.A = (3 * sd.A) / 5 ∧
    ld.B = (3 * sd.B) / 20 ∧
    ld.C = (3 * sd.C) / 10 ∧
    ld.D = sd.A / 3 := by
  sorry

end NUMINAMATH_CALUDE_lunch_packing_days_l1887_188707


namespace NUMINAMATH_CALUDE_multiple_of_all_positive_integers_l1887_188792

theorem multiple_of_all_positive_integers (n : ℤ) : 
  (∀ m : ℕ+, ∃ k : ℤ, n = k * m) ↔ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_all_positive_integers_l1887_188792


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l1887_188747

theorem absolute_value_sum_zero (a b : ℝ) :
  |a - 3| + |b + 6| = 0 → (a + b - 2 = -5 ∧ a - b - 2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l1887_188747


namespace NUMINAMATH_CALUDE_base_8_4512_equals_2378_l1887_188700

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4512_equals_2378 :
  base_8_to_10 [2, 1, 5, 4] = 2378 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4512_equals_2378_l1887_188700


namespace NUMINAMATH_CALUDE_discounted_price_per_shirt_l1887_188743

-- Define the given conditions
def number_of_shirts : ℕ := 3
def original_total_price : ℚ := 60
def discount_percentage : ℚ := 40

-- Define the theorem
theorem discounted_price_per_shirt :
  let discount_amount : ℚ := (discount_percentage / 100) * original_total_price
  let sale_price : ℚ := original_total_price - discount_amount
  let price_per_shirt : ℚ := sale_price / number_of_shirts
  price_per_shirt = 12 := by sorry

end NUMINAMATH_CALUDE_discounted_price_per_shirt_l1887_188743


namespace NUMINAMATH_CALUDE_time_after_316h59m59s_l1887_188721

/-- Represents time on a 12-hour digital clock -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

def addTime (t : Time) (hours minutes seconds : Nat) : Time :=
  let totalSeconds := t.hours * 3600 + t.minutes * 60 + t.seconds + hours * 3600 + minutes * 60 + seconds
  let newHours := (totalSeconds / 3600) % 12
  let newMinutes := (totalSeconds % 3600) / 60
  let newSeconds := totalSeconds % 60
  { hours := if newHours = 0 then 12 else newHours, minutes := newMinutes, seconds := newSeconds }

def sumDigits (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_after_316h59m59s (startTime : Time) :
  startTime.hours = 3 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 →
  sumDigits (addTime startTime 316 59 59) = 125 := by
  sorry

end NUMINAMATH_CALUDE_time_after_316h59m59s_l1887_188721


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1887_188737

theorem units_digit_of_expression : ∃ n : ℕ, (9 * 19 * 1989 - 9^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1887_188737


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1887_188720

theorem subtraction_of_decimals : 3.75 - 0.48 = 3.27 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1887_188720


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1887_188757

theorem quadratic_rewrite (b : ℝ) (h1 : b < 0) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 1/4 = (x + m)^2 + 1/6) →
  b = -1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1887_188757


namespace NUMINAMATH_CALUDE_min_value_theorem_l1887_188710

/-- The minimum value of a specific function given certain conditions -/
theorem min_value_theorem (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  (h1 : c * y + b * z = a) 
  (h2 : a * z + c * x = b) 
  (h3 : b * x + a * y = c) : 
  (∃ (m : ℝ), m = (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)) ∧ 
   ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
   x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ m) ∧ 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1887_188710


namespace NUMINAMATH_CALUDE_solve_system_l1887_188730

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1887_188730


namespace NUMINAMATH_CALUDE_current_speed_l1887_188713

theorem current_speed (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ c : ℝ, c = 16 / 7 ∧ 
    (boat_speed - c) * upstream_time = (boat_speed + c) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l1887_188713


namespace NUMINAMATH_CALUDE_abc_theorem_l1887_188738

theorem abc_theorem (a b c : ℕ+) (x y z w : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c)
  (h_eq : (a : ℝ) ^ x = (b : ℝ) ^ y ∧ (b : ℝ) ^ y = (c : ℝ) ^ z ∧ (c : ℝ) ^ z = 70 ^ w)
  (h_sum : 1 / x + 1 / y + 1 / z = 1 / w) :
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_abc_theorem_l1887_188738


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1887_188724

theorem smallest_prime_divisor_of_sum : ∃ (p : ℕ), p.Prime ∧ p ∣ (3^19 + 11^23) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^19 + 11^23) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1887_188724


namespace NUMINAMATH_CALUDE_total_fallen_blocks_l1887_188751

/-- Represents the heights of three stacks of blocks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of fallen blocks -/
def fallen_blocks (stacks : BlockStacks) (standing_second standing_third : ℕ) : ℕ :=
  stacks.first + (stacks.second - standing_second) + (stacks.third - standing_third)

theorem total_fallen_blocks : 
  let stacks : BlockStacks := { 
    first := 7, 
    second := 7 + 5, 
    third := 7 + 5 + 7 
  }
  fallen_blocks stacks 2 3 = 33 := by
  sorry

#eval fallen_blocks { first := 7, second := 7 + 5, third := 7 + 5 + 7 } 2 3

end NUMINAMATH_CALUDE_total_fallen_blocks_l1887_188751


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l1887_188750

/-- Represents the heights of cuts in the cube -/
structure CutHeights where
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ

/-- Calculates the total surface area of a stacked solid formed from a cube -/
def totalSurfaceArea (cubeSideLength : ℝ) (cuts : CutHeights) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific cut and stacked cube -/
theorem specific_cube_surface_area :
  let cubeSideLength : ℝ := 2
  let cuts : CutHeights := { h1 := 1/4, h2 := 1/4 + 1/5, h3 := 1/4 + 1/5 + 1/8 }
  totalSurfaceArea cubeSideLength cuts = 12 := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_surface_area_l1887_188750


namespace NUMINAMATH_CALUDE_radical_axes_theorem_l1887_188723

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The radical axis of two circles --/
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 - c1.radius^2 = 
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 - c2.radius^2}

/-- Three lines are either coincident, parallel, or concurrent --/
def linesCoincidentParallelOrConcurrent (l1 l2 l3 : Set (ℝ × ℝ)) : Prop :=
  (l1 = l2 ∧ l2 = l3) ∨ 
  (∀ p1 ∈ l1, ∀ p2 ∈ l2, ∀ p3 ∈ l3, 
    (p1.1 - p2.1) * (p3.2 - p2.2) = (p3.1 - p2.1) * (p1.2 - p2.2)) ∨
  (∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3)

/-- The Theorem of Radical Axes --/
theorem radical_axes_theorem (c1 c2 c3 : Circle) :
  linesCoincidentParallelOrConcurrent 
    (radicalAxis c1 c2) 
    (radicalAxis c2 c3) 
    (radicalAxis c3 c1) :=
sorry

end NUMINAMATH_CALUDE_radical_axes_theorem_l1887_188723


namespace NUMINAMATH_CALUDE_max_volume_cone_l1887_188785

/-- Given a right-angled triangle with hypotenuse c, the triangle that forms a cone
    with maximum volume when rotated around one of its legs has the following properties: -/
theorem max_volume_cone (c : ℝ) (h : c > 0) :
  ∃ (x y : ℝ),
    -- The triangle is right-angled
    x^2 + y^2 = c^2 ∧
    -- x and y are positive
    x > 0 ∧ y > 0 ∧
    -- y is the optimal radius of the cone's base
    y = c * Real.sqrt (2/3) ∧
    -- x is the optimal height of the cone
    x = c / Real.sqrt 3 ∧
    -- The volume formed by this triangle is maximum
    ∀ (x' y' : ℝ), x'^2 + y'^2 = c^2 → x' > 0 → y' > 0 →
      (1/3) * π * y'^2 * x' ≤ (1/3) * π * y^2 * x ∧
    -- The maximum volume is (2 * π * √3 * c^3) / 27
    (1/3) * π * y^2 * x = (2 * π * Real.sqrt 3 * c^3) / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_cone_l1887_188785


namespace NUMINAMATH_CALUDE_max_sequence_length_l1887_188790

def is_valid_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i + 4 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) < 0) ∧
  (∀ i : ℕ, i + 8 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8)) > 0)

theorem max_sequence_length :
  (∃ (a : ℕ → ℝ), is_valid_sequence a 12) ∧
  (∀ (a : ℕ → ℝ) (n : ℕ), n > 12 → ¬ is_valid_sequence a n) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l1887_188790


namespace NUMINAMATH_CALUDE_tournament_result_l1887_188783

/-- The number of athletes with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * (m.choose k)

/-- The number of athletes with 4 points after 7 rounds in a tournament with 2^n + 6 participants -/
def athletes_with_four_points (n : ℕ) : ℕ := 35 * 2^(n - 7) + 2

theorem tournament_result (n : ℕ) (h : n > 7) :
  athletes_with_four_points n = f n 7 4 + 2 := by sorry

end NUMINAMATH_CALUDE_tournament_result_l1887_188783


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l1887_188789

/-- Calculates the total cost of installing two types of fences around a rectangular field -/
theorem fence_cost_calculation (length width : ℝ) (barbed_wire_cost picket_fence_cost : ℝ) 
  (num_gates gate_width : ℝ) : 
  length = 500 ∧ 
  width = 150 ∧ 
  barbed_wire_cost = 1.2 ∧ 
  picket_fence_cost = 2.5 ∧ 
  num_gates = 4 ∧ 
  gate_width = 1.25 → 
  (2 * (length + width) - num_gates * gate_width) * barbed_wire_cost + 
  2 * (length + width) * picket_fence_cost = 4804 := by
  sorry


end NUMINAMATH_CALUDE_fence_cost_calculation_l1887_188789


namespace NUMINAMATH_CALUDE_determinant_equality_l1887_188719

theorem determinant_equality (p q r s : ℝ) : 
  p * s - q * r = 10 → (p + 2*r) * s - (q + 2*s) * r = 10 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1887_188719


namespace NUMINAMATH_CALUDE_suresh_completion_time_l1887_188766

/-- Proves that Suresh can complete the job alone in 15 hours given the problem conditions -/
theorem suresh_completion_time :
  ∀ (S : ℝ),
  (S > 0) →  -- Suresh's completion time is positive
  (9 / S + 10 / 25 = 1) →  -- Combined work of Suresh and Ashutosh equals the whole job
  (S = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_suresh_completion_time_l1887_188766


namespace NUMINAMATH_CALUDE_smallest_m_equals_n_l1887_188715

theorem smallest_m_equals_n (n : ℕ) (hn : n > 1) :
  ∃ (m : ℕ),
    (∀ (a b : ℕ) (ha : a ∈ Finset.range (2 * n)) (hb : b ∈ Finset.range (2 * n)) (hab : a ≠ b),
      ∃ (x y : ℕ) (hxy : x + y > 0),
        (2 * n ∣ a * x + b * y) ∧ (x + y ≤ m)) ∧
    (∀ (k : ℕ),
      (∀ (a b : ℕ) (ha : a ∈ Finset.range (2 * n)) (hb : b ∈ Finset.range (2 * n)) (hab : a ≠ b),
        ∃ (x y : ℕ) (hxy : x + y > 0),
          (2 * n ∣ a * x + b * y) ∧ (x + y ≤ k)) →
      k ≥ m) ∧
    m = n :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_equals_n_l1887_188715


namespace NUMINAMATH_CALUDE_train_speed_l1887_188714

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 4) :
  length / time = 40 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1887_188714


namespace NUMINAMATH_CALUDE_smallest_cube_root_plus_small_fraction_l1887_188722

theorem smallest_cube_root_plus_small_fraction (m n : ℕ) (r : ℝ) : 
  (m > 0) →
  (n > 0) →
  (r > 0) →
  (r < 1/500) →
  (m : ℝ)^(1/3) = n + r →
  (∀ m' n' r', m' > 0 → n' > 0 → r' > 0 → r' < 1/500 → (m' : ℝ)^(1/3) = n' + r' → m' ≥ m) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_plus_small_fraction_l1887_188722


namespace NUMINAMATH_CALUDE_unpartnered_students_count_l1887_188760

/-- Represents the number of students in a class -/
structure ClassCount where
  males : ℕ
  females : ℕ

/-- The number of students unable to partner with the opposite gender -/
def unpartnered_students (classes : List ClassCount) : ℕ :=
  let total_males := classes.map (·.males) |>.sum
  let total_females := classes.map (·.females) |>.sum
  Int.natAbs (total_males - total_females)

/-- The main theorem stating the number of unpartnered students -/
theorem unpartnered_students_count : 
  let classes : List ClassCount := [
    ⟨18, 12⟩,  -- First 6th grade class
    ⟨16, 20⟩,  -- Second 6th grade class
    ⟨13, 19⟩,  -- Third 6th grade class
    ⟨23, 21⟩   -- 7th grade class
  ]
  unpartnered_students classes = 2 := by
  sorry

end NUMINAMATH_CALUDE_unpartnered_students_count_l1887_188760


namespace NUMINAMATH_CALUDE_y_value_at_16_l1887_188776

/-- Given a function y = k * x^(1/4) where y = 3 * √3 when x = 9, 
    prove that y = 6 when x = 16 -/
theorem y_value_at_16 (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/4)) →
  y 9 = 3 * Real.sqrt 3 →
  y 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_16_l1887_188776


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1887_188773

theorem necessary_but_not_sufficient : 
  (∀ x y : ℝ, x > 3 ∧ y ≥ 3 → x^2 + y^2 ≥ 9) ∧ 
  (∃ x y : ℝ, x^2 + y^2 ≥ 9 ∧ ¬(x > 3 ∧ y ≥ 3)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1887_188773


namespace NUMINAMATH_CALUDE_triangle_type_indeterminate_l1887_188761

theorem triangle_type_indeterminate (A B C : ℝ) 
  (triangle_sum : A + B + C = π) 
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (inequality : Real.sin A * Real.sin C > Real.cos A * Real.cos C) : 
  ¬(∀ α : ℝ, (0 < α ∧ α < π) → 
    ((A < π/2 ∧ B < π/2 ∧ C < π/2) ∨ 
     (A = π/2 ∨ B = π/2 ∨ C = π/2) ∨ 
     (A > π/2 ∨ B > π/2 ∨ C > π/2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_type_indeterminate_l1887_188761


namespace NUMINAMATH_CALUDE_two_digit_average_decimal_l1887_188734

theorem two_digit_average_decimal (m n : ℕ) : 
  (10 ≤ m ∧ m < 100) →
  (10 ≤ n ∧ n < 100) →
  (m + n) / 2 = m + n / 10 →
  m = n :=
by sorry

end NUMINAMATH_CALUDE_two_digit_average_decimal_l1887_188734


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_similarity_theorem_l1887_188771

-- Define cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define similarity of quadrilaterals
def are_similar_quadrilaterals (A B C D A' B' C' D' : Point) : Prop := sorry

-- Define area of a triangle
def area_triangle (A B C : Point) : ℝ := sorry

-- Define distance between two points
def distance (A B : Point) : ℝ := sorry

theorem cyclic_quadrilateral_similarity_theorem 
  (A B C D A' B' C' D' : Point) 
  (h1 : is_cyclic_quadrilateral A B C D) 
  (h2 : is_cyclic_quadrilateral A' B' C' D')
  (h3 : are_similar_quadrilaterals A B C D A' B' C' D') :
  (distance A A')^2 * area_triangle B C D + (distance C C')^2 * area_triangle A B D = 
  (distance B B')^2 * area_triangle A C D + (distance D D')^2 * area_triangle A B C := by
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_similarity_theorem_l1887_188771


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l1887_188731

theorem rachel_reading_homework (literature_pages : ℕ) (additional_reading_pages : ℕ) 
  (h1 : literature_pages = 10) 
  (h2 : additional_reading_pages = 6) : 
  literature_pages + additional_reading_pages = 16 := by
  sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l1887_188731


namespace NUMINAMATH_CALUDE_celine_change_l1887_188746

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine buys -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine buys -/
def smartphones_bought : ℕ := 4

/-- The amount of money Celine has in dollars -/
def money_available : ℕ := 3000

/-- The change Celine receives after her purchase -/
theorem celine_change : 
  money_available - (laptop_price * laptops_bought + smartphone_price * smartphones_bought) = 200 := by
  sorry

end NUMINAMATH_CALUDE_celine_change_l1887_188746


namespace NUMINAMATH_CALUDE_circle_diameter_l1887_188772

theorem circle_diameter (A : Real) (r : Real) (d : Real) : 
  A = Real.pi * r^2 → A = 64 * Real.pi → d = 2 * r → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l1887_188772


namespace NUMINAMATH_CALUDE_factorial_fraction_l1887_188779

theorem factorial_fraction (N : ℕ) : 
  (Nat.factorial (N + 1) * (N + 2)) / Nat.factorial (N + 3) = 1 / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_l1887_188779


namespace NUMINAMATH_CALUDE_jane_albert_same_committee_l1887_188755

/-- The number of MBAs --/
def n : ℕ := 6

/-- The number of members in each committee --/
def k : ℕ := 3

/-- The number of committees to be formed --/
def num_committees : ℕ := 2

/-- The total number of ways to form the committees --/
def total_ways : ℕ := Nat.choose n k

/-- The number of ways Jane and Albert can be on the same committee --/
def favorable_ways : ℕ := Nat.choose (n - 2) (k - 2)

/-- The probability that Jane and Albert are on the same committee --/
def prob_same_committee : ℚ := favorable_ways / total_ways

theorem jane_albert_same_committee :
  prob_same_committee = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_jane_albert_same_committee_l1887_188755


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l1887_188729

theorem quadratic_polynomial_problem :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (20/9) * x^2 + (20/3) * x - 40) ∧
    p (-6) = 0 ∧
    p 3 = 0 ∧
    p (-3) = -40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l1887_188729


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1887_188797

theorem trigonometric_identity (α : ℝ) :
  (Real.cos (4 * α - 3 * Real.pi) ^ 2 - 4 * Real.cos (2 * α - Real.pi) ^ 2 + 3) /
  (Real.cos (4 * α + 3 * Real.pi) ^ 2 + 4 * Real.cos (2 * α + Real.pi) ^ 2 - 1) =
  Real.tan (2 * α) ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1887_188797


namespace NUMINAMATH_CALUDE_quadratic_min_values_l1887_188733

-- Define the quadratic function
def f (x a : ℝ) : ℝ := 2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

-- State the theorem
theorem quadratic_min_values (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x a ≥ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f x a = 2) →
  a = 0 ∨ a = 2 ∨ a = -3 - Real.sqrt 7 ∨ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_min_values_l1887_188733


namespace NUMINAMATH_CALUDE_infinitely_many_square_averages_l1887_188704

theorem infinitely_many_square_averages :
  ∃ f : ℕ → ℕ, 
    (f 0 = 1) ∧ 
    (∀ k : ℕ, f k < f (k + 1)) ∧
    (∀ k : ℕ, ∃ m : ℕ, (f k * (f k + 1) * (2 * f k + 1)) / 6 = m^2 * f k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_square_averages_l1887_188704


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1887_188768

theorem fraction_equals_zero (x : ℝ) (h : x = 1) : (2 * x - 2) / (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1887_188768


namespace NUMINAMATH_CALUDE_max_ab_bisecting_line_l1887_188727

/-- A line that bisects the circumference of a circle --/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects : ∀ (x y : ℝ), a * x + b * y - 1 = 0 → x^2 + y^2 - 4*x - 4*y - 8 = 0

/-- The maximum value of ab for a bisecting line --/
theorem max_ab_bisecting_line (l : BisectingLine) : 
  ∃ (max : ℝ), (∀ (l' : BisectingLine), l'.a * l'.b ≤ max) ∧ max = 1/16 := by
sorry

end NUMINAMATH_CALUDE_max_ab_bisecting_line_l1887_188727


namespace NUMINAMATH_CALUDE_model_price_increase_l1887_188725

theorem model_price_increase (original_price : ℚ) (original_quantity : ℕ) (new_quantity : ℕ) 
  (h1 : original_price = 45 / 100)
  (h2 : original_quantity = 30)
  (h3 : new_quantity = 27) :
  let total_saved := original_price * original_quantity
  let new_price := total_saved / new_quantity
  new_price = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_model_price_increase_l1887_188725


namespace NUMINAMATH_CALUDE_shadow_boundary_equation_l1887_188767

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The equation of the shadow boundary on the xy-plane -/
def shadowBoundary (s : Sphere) (lightSource : Point3D) : ℝ → ℝ := fun x ↦ -14

/-- Theorem: The shadow boundary of the given sphere with the given light source is y = -14 -/
theorem shadow_boundary_equation (s : Sphere) (lightSource : Point3D) :
  s.center = Point3D.mk 2 0 2 →
  s.radius = 2 →
  lightSource = Point3D.mk 2 (-2) 6 →
  ∀ x : ℝ, shadowBoundary s lightSource x = -14 := by
  sorry

#check shadow_boundary_equation

end NUMINAMATH_CALUDE_shadow_boundary_equation_l1887_188767


namespace NUMINAMATH_CALUDE_paul_bought_101_books_l1887_188770

/-- Calculates the number of books bought given initial and final book counts -/
def books_bought (initial_count final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Proves that Paul bought 101 books -/
theorem paul_bought_101_books (initial_count final_count : ℕ) 
  (h1 : initial_count = 50)
  (h2 : final_count = 151) :
  books_bought initial_count final_count = 101 := by
  sorry

end NUMINAMATH_CALUDE_paul_bought_101_books_l1887_188770


namespace NUMINAMATH_CALUDE_custom_mult_factorial_difference_l1887_188793

-- Define the custom multiplication operation
def custom_mult (a b : ℕ) : ℕ := a * b + a + b

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the chained custom multiplication
def chained_custom_mult (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => custom_mult (chained_custom_mult n) (n + 1)

theorem custom_mult_factorial_difference :
  factorial 10 - chained_custom_mult 9 = 1 := by
  sorry


end NUMINAMATH_CALUDE_custom_mult_factorial_difference_l1887_188793


namespace NUMINAMATH_CALUDE_dot_product_perpendiculars_l1887_188711

/-- Given a point P(x₀, y₀) on the curve y = x + 2/x for x > 0,
    and points A and B as the feet of perpendiculars from P to y = x and y-axis respectively,
    prove that the dot product of PA and PB is -1. -/
theorem dot_product_perpendiculars (x₀ : ℝ) (h₀ : x₀ > 0) : 
  let y₀ := x₀ + 2 / x₀
  let P := (x₀, y₀)
  let A := ((x₀ + y₀) / 2, (x₀ + y₀) / 2)  -- Foot of perpendicular to y = x
  let B := (0, y₀)  -- Foot of perpendicular to y-axis
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_perpendiculars_l1887_188711


namespace NUMINAMATH_CALUDE_table_football_points_l1887_188765

/-- The total points scored by four friends in table football games -/
def total_points (darius matt marius sofia : ℕ) : ℕ :=
  darius + matt + marius + sofia

/-- Theorem stating the total points scored by the four friends -/
theorem table_football_points : ∃ (darius matt marius sofia : ℕ),
  darius = 10 ∧
  marius = darius + 3 ∧
  matt = darius + 5 ∧
  sofia = 2 * matt ∧
  total_points darius matt marius sofia = 68 := by
  sorry


end NUMINAMATH_CALUDE_table_football_points_l1887_188765


namespace NUMINAMATH_CALUDE_initial_trees_per_row_garden_problem_l1887_188728

theorem initial_trees_per_row : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_rows added_rows final_trees_per_row result =>
    let final_rows := initial_rows + added_rows
    (initial_rows * result = final_rows * final_trees_per_row) →
    (result = 42)

/-- Given the initial number of rows, added rows, and final trees per row,
    prove that the initial number of trees per row is 42. -/
theorem garden_problem (initial_rows added_rows final_trees_per_row : ℕ)
    (h1 : initial_rows = 24)
    (h2 : added_rows = 12)
    (h3 : final_trees_per_row = 28) :
    initial_trees_per_row initial_rows added_rows final_trees_per_row 42 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_per_row_garden_problem_l1887_188728


namespace NUMINAMATH_CALUDE_post_office_problem_l1887_188798

/-- Proves that given the conditions from the post office problem, each month has 30 days. -/
theorem post_office_problem (letters_per_day : ℕ) (packages_per_day : ℕ) 
  (total_mail : ℕ) (num_months : ℕ) :
  letters_per_day = 60 →
  packages_per_day = 20 →
  total_mail = 14400 →
  num_months = 6 →
  (total_mail / (letters_per_day + packages_per_day)) / num_months = 30 := by
  sorry

end NUMINAMATH_CALUDE_post_office_problem_l1887_188798


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1887_188718

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x ≠ 0) ∧ (∃ x, x ≠ 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1887_188718


namespace NUMINAMATH_CALUDE_existence_of_finite_sequence_no_infinite_sequence_l1887_188702

/-- S(k) denotes the sum of all digits of a positive integer k in its decimal representation. -/
def S (k : ℕ+) : ℕ :=
  sorry

/-- For any positive integer n, there exists an arithmetic sequence of positive integers
    a₁, a₂, ..., aₙ such that S(a₁) < S(a₂) < ... < S(aₙ). -/
theorem existence_of_finite_sequence (n : ℕ+) :
  ∃ (a : ℕ+ → ℕ+) (d : ℕ+), (∀ i : ℕ+, i ≤ n → a i = a 1 + (i - 1) * d) ∧
    (∀ i : ℕ+, i < n → S (a i) < S (a (i + 1))) :=
  sorry

/-- There does not exist an infinite arithmetic sequence of positive integers {aₙ}
    such that S(a₁) < S(a₂) < ... < S(aₙ) < ... -/
theorem no_infinite_sequence :
  ¬ ∃ (a : ℕ+ → ℕ+) (d : ℕ+), (∀ i : ℕ+, a i = a 1 + (i - 1) * d) ∧
    (∀ i j : ℕ+, i < j → S (a i) < S (a j)) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_finite_sequence_no_infinite_sequence_l1887_188702


namespace NUMINAMATH_CALUDE_tank_water_fraction_l1887_188735

theorem tank_water_fraction (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) : 
  tank_capacity = 56 →
  initial_fraction = 3/4 →
  added_water = 7 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7/8 := by
sorry

end NUMINAMATH_CALUDE_tank_water_fraction_l1887_188735


namespace NUMINAMATH_CALUDE_triangle_solution_l1887_188791

theorem triangle_solution (c t r : ℝ) (hc : c = 30) (ht : t = 336) (hr : r = 8) :
  ∃ (a b : ℝ),
    a + b + c = 2 * (t / r) ∧
    t = r * (a + b + c) / 2 ∧
    t^2 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) ∧
    a = 26 ∧
    b = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_solution_l1887_188791


namespace NUMINAMATH_CALUDE_gray_area_is_65_l1887_188748

/-- Given two overlapping rectangles, calculates the area of the gray part -/
def gray_area (width1 length1 width2 length2 black_area : ℕ) : ℕ :=
  width2 * length2 - (width1 * length1 - black_area)

/-- Theorem stating that the area of the gray part is 65 -/
theorem gray_area_is_65 :
  gray_area 8 10 12 9 37 = 65 := by
  sorry

end NUMINAMATH_CALUDE_gray_area_is_65_l1887_188748


namespace NUMINAMATH_CALUDE_equipment_production_calculation_l1887_188781

/-- Given a total production and a sample with known quantities from two equipment types,
    calculate the total production of the second equipment type. -/
theorem equipment_production_calculation
  (total_production : ℕ) -- Total number of pieces produced
  (sample_size : ℕ) -- Size of the sample
  (sample_a : ℕ) -- Number of pieces from equipment A in the sample
  (h1 : total_production = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_a = 50)
  : ∃ (total_b : ℕ), total_b = 1800 ∧ total_b + (total_production - total_b) = total_production :=
by
  sorry

#check equipment_production_calculation

end NUMINAMATH_CALUDE_equipment_production_calculation_l1887_188781


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1887_188744

/-- 
Given two bases a and b, both greater than 2, 
this function returns the base-10 representation of 21 in base a
-/
def base_a_to_10 (a : ℕ) : ℕ := 2 * a + 1

/-- 
Given two bases a and b, both greater than 2, 
this function returns the base-10 representation of 12 in base b
-/
def base_b_to_10 (b : ℕ) : ℕ := b + 2

/--
This theorem states that 7 is the smallest base-10 integer that can be represented
as 21_a in one base and 12_b in a different base, where a and b are any bases larger than 2.
-/
theorem smallest_dual_base_representation :
  ∀ a b : ℕ, a > 2 → b > 2 →
  (∃ n : ℕ, n < 7 ∧ base_a_to_10 a = n ∧ base_b_to_10 b = n) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1887_188744
