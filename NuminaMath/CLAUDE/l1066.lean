import Mathlib

namespace NUMINAMATH_CALUDE_power_comparison_l1066_106601

theorem power_comparison : 2^444 = 4^222 ∧ 2^444 < 3^333 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l1066_106601


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l1066_106693

/-- The measure of one interior angle of a regular octagon in degrees -/
def regular_octagon_interior_angle : ℝ := 135

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l1066_106693


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l1066_106688

/-- Given a quadratic equation 2x^2 = 7x - 5, prove that when converted to the general form
    ax^2 + bx + c = 0, the coefficient of the linear term (b) is -7 and the constant term (c) is 5 -/
theorem quadratic_equation_conversion :
  ∃ (a b c : ℝ), (∀ x, 2 * x^2 = 7 * x - 5) →
  (∀ x, a * x^2 + b * x + c = 0) ∧ b = -7 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l1066_106688


namespace NUMINAMATH_CALUDE_two_fifths_of_seven_point_five_l1066_106626

theorem two_fifths_of_seven_point_five : (2 / 5 : ℚ) * (15 / 2 : ℚ) = (3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_seven_point_five_l1066_106626


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l1066_106651

theorem inscribed_squares_segment_product :
  ∀ (a b : ℝ),
    (∃ (inner_area outer_area : ℝ),
      inner_area = 16 ∧
      outer_area = 18 ∧
      (∃ (inner_side outer_side : ℝ),
        inner_side^2 = inner_area ∧
        outer_side^2 = outer_area ∧
        a + b = outer_side ∧
        (a^2 + b^2) = inner_side^2)) →
    a * b = -7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l1066_106651


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1066_106631

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h1 : a 4 + a 8 = -2) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1066_106631


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1066_106668

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x^2 = a - 1) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1066_106668


namespace NUMINAMATH_CALUDE_t_leq_s_l1066_106682

theorem t_leq_s (a b t s : ℝ) (ht : t = a + 2*b) (hs : s = a + b^2 + 1) : t ≤ s := by
  sorry

end NUMINAMATH_CALUDE_t_leq_s_l1066_106682


namespace NUMINAMATH_CALUDE_tangency_line_parallel_to_common_tangent_l1066_106603

/-- Given three parabolas p₁, p₂, and p₃, where p₁ and p₂ both touch p₃,
    the line connecting the points of tangency of p₁ and p₂ with p₃
    is parallel to the common tangent of p₁ and p₂. -/
theorem tangency_line_parallel_to_common_tangent
  (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let p₁ := fun x => -x^2 + b₁ * x + c₁
  let p₂ := fun x => -x^2 + b₂ * x + c₂
  let p₃ := fun x => x^2 + b₃ * x + c₃
  let x₁ := (b₁ - b₃) / 4
  let y₁ := p₃ x₁
  let x₂ := (b₂ - b₃) / 4
  let y₂ := p₃ x₂
  let m_tangency := (y₂ - y₁) / (x₂ - x₁)
  let m_common_tangent := (4 * (c₁ - c₂) - 2 * b₃ * (b₁ - b₂)) / (2 * (b₂ - b₁))
  (b₃ - b₁)^2 = 8 * (c₃ - c₁) →
  (b₃ - b₂)^2 = 8 * (c₃ - c₂) →
  m_tangency = m_common_tangent :=
by sorry

end NUMINAMATH_CALUDE_tangency_line_parallel_to_common_tangent_l1066_106603


namespace NUMINAMATH_CALUDE_lost_ship_depth_l1066_106658

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def ship_depth (descent_rate : ℝ) (time_taken : ℝ) : ℝ := descent_rate * time_taken

/-- Theorem stating the depth of the lost ship -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 80
  let time_taken : ℝ := 50
  ship_depth descent_rate time_taken = 4000 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l1066_106658


namespace NUMINAMATH_CALUDE_bake_sale_cookies_l1066_106630

/-- The number of chocolate chip cookies Jenny brought to the bake sale -/
def jenny_chocolate_chip : ℕ := 50

/-- The total number of peanut butter cookies at the bake sale -/
def total_peanut_butter : ℕ := 70

/-- The number of lemon cookies Marcus brought to the bake sale -/
def marcus_lemon : ℕ := 20

/-- The probability of picking a peanut butter cookie -/
def prob_peanut_butter : ℚ := 1/2

theorem bake_sale_cookies :
  jenny_chocolate_chip = 50 ∧
  total_peanut_butter = 70 ∧
  marcus_lemon = 20 ∧
  prob_peanut_butter = 1/2 →
  jenny_chocolate_chip + marcus_lemon = total_peanut_butter :=
by sorry

end NUMINAMATH_CALUDE_bake_sale_cookies_l1066_106630


namespace NUMINAMATH_CALUDE_probability_twelve_rolls_eight_sided_die_l1066_106677

/-- The probability of rolling an eight-sided die 12 times, where the first 11 rolls are all
    different from their immediate predecessors, and the 12th roll matches the 11th roll. -/
def probability_twelve_rolls (n : ℕ) : ℚ :=
  if n = 8 then
    (7 : ℚ)^10 / 8^11
  else
    0

/-- Theorem stating that the probability of the described event with an eight-sided die
    is equal to 7^10 / 8^11. -/
theorem probability_twelve_rolls_eight_sided_die :
  probability_twelve_rolls 8 = (7 : ℚ)^10 / 8^11 :=
by sorry

end NUMINAMATH_CALUDE_probability_twelve_rolls_eight_sided_die_l1066_106677


namespace NUMINAMATH_CALUDE_f_nonnegative_l1066_106678

/-- Definition of the function f --/
def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) + B * (a^2*b + b^2*c + c^2*a + a*b^2 + b*c^2 + c*a^2) + C * a * b * c

/-- Triangle inequality --/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem --/
theorem f_nonnegative (A B C : ℝ) :
  (f A B C 1 1 1 ≥ 0) →
  (f A B C 1 1 0 ≥ 0) →
  (f A B C 2 1 1 ≥ 0) →
  ∀ a b c : ℝ, is_triangle a b c → f A B C a b c ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_l1066_106678


namespace NUMINAMATH_CALUDE_dima_places_more_berries_l1066_106675

/-- The total number of berries on the bush -/
def total_berries : ℕ := 450

/-- Dima's picking pattern: fraction of berries that go into the basket -/
def dima_basket_ratio : ℚ := 1/2

/-- Sergei's picking pattern: fraction of berries that go into the basket -/
def sergei_basket_ratio : ℚ := 2/3

/-- Dima's picking speed relative to Sergei -/
def dima_speed_ratio : ℕ := 2

/-- The number of berries Dima puts in the basket -/
def dima_basket_berries : ℕ := 150

/-- The number of berries Sergei puts in the basket -/
def sergei_basket_berries : ℕ := 100

/-- Theorem stating that Dima places 50 more berries into the basket than Sergei -/
theorem dima_places_more_berries :
  dima_basket_berries - sergei_basket_berries = 50 :=
sorry

end NUMINAMATH_CALUDE_dima_places_more_berries_l1066_106675


namespace NUMINAMATH_CALUDE_quadrilateral_interior_point_angles_l1066_106618

theorem quadrilateral_interior_point_angles 
  (a b c d x y z w : ℝ) 
  (h1 : a = x + y / 2)
  (h2 : b = y + z / 2)
  (h3 : c = z + w / 2)
  (h4 : d = w + x / 2)
  (h5 : x + y + z + w = 360) :
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_interior_point_angles_l1066_106618


namespace NUMINAMATH_CALUDE_students_playing_neither_l1066_106664

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 36)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l1066_106664


namespace NUMINAMATH_CALUDE_paramEquations_represent_line_l1066_106647

/-- Parametric equations of a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line y = 2x + 1 -/
def linearEquation (x y : ℝ) : Prop := y = 2 * x + 1

/-- The parametric equations x = t - 1 and y = 2t - 1 -/
def paramEquations : ParametricLine :=
  { x := fun t => t - 1
    y := fun t => 2 * t - 1 }

/-- Theorem: The parametric equations represent the line y = 2x + 1 -/
theorem paramEquations_represent_line :
  ∀ t : ℝ, linearEquation (paramEquations.x t) (paramEquations.y t) := by
  sorry


end NUMINAMATH_CALUDE_paramEquations_represent_line_l1066_106647


namespace NUMINAMATH_CALUDE_soccer_ball_price_is_40_l1066_106684

def soccer_ball_price (total_balls : ℕ) (amount_given : ℕ) (change_received : ℕ) : ℕ :=
  (amount_given - change_received) / total_balls

theorem soccer_ball_price_is_40 :
  soccer_ball_price 2 100 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_price_is_40_l1066_106684


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l1066_106600

/-- Proves that the percentage of passed candidates is 32% given the conditions of the examination --/
theorem exam_pass_percentage : 
  ∀ (total_candidates : ℕ) 
    (num_girls : ℕ) 
    (num_boys : ℕ) 
    (fail_percentage : ℝ) 
    (pass_percentage : ℝ),
  total_candidates = 2000 →
  num_girls = 900 →
  num_boys = total_candidates - num_girls →
  fail_percentage = 68 →
  pass_percentage = 100 - fail_percentage →
  pass_percentage = 32 := by
sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l1066_106600


namespace NUMINAMATH_CALUDE_solve_coloring_book_problem_l1066_106643

def coloring_book_problem (book1 book2 book3 colored : ℕ) : Prop :=
  let total := book1 + book2 + book3
  total - colored = 53

theorem solve_coloring_book_problem :
  coloring_book_problem 35 45 40 67 := by
  sorry

end NUMINAMATH_CALUDE_solve_coloring_book_problem_l1066_106643


namespace NUMINAMATH_CALUDE_inequality_proof_l1066_106612

theorem inequality_proof (a b c : ℝ) : 
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1066_106612


namespace NUMINAMATH_CALUDE_inequality_proof_l1066_106663

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1066_106663


namespace NUMINAMATH_CALUDE_valid_permutations_count_l1066_106661

/-- Represents a permutation of 8 people around a circular table. -/
def CircularPermutation := Fin 8 → Fin 8

/-- Checks if two positions are adjacent on a circular table with 8 positions. -/
def is_adjacent (a b : Fin 8) : Prop :=
  (a - b) % 8 = 1 ∨ (b - a) % 8 = 1

/-- Checks if two positions are opposite on a circular table with 8 positions. -/
def is_opposite (a b : Fin 8) : Prop :=
  (a - b) % 8 = 4

/-- Checks if a permutation is valid according to the problem conditions. -/
def is_valid_permutation (p : CircularPermutation) : Prop :=
  ∀ i : Fin 8, 
    p i ≠ i ∧ 
    ¬is_adjacent i (p i) ∧ 
    ¬is_opposite i (p i)

/-- The main theorem stating that there are exactly 3 valid permutations. -/
theorem valid_permutations_count :
  ∃! (perms : Finset CircularPermutation),
    (∀ p ∈ perms, is_valid_permutation p) ∧
    perms.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valid_permutations_count_l1066_106661


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1066_106624

theorem sin_150_degrees : Real.sin (150 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1066_106624


namespace NUMINAMATH_CALUDE_not_q_sufficient_not_necessary_for_p_l1066_106680

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ 1
def q (x : ℝ) : Prop := 1/x < 1

-- Theorem stating that ¬q is a sufficient but not necessary condition for p
theorem not_q_sufficient_not_necessary_for_p :
  (∀ x : ℝ, ¬(q x) → p x) ∧ 
  (∃ x : ℝ, p x ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_not_q_sufficient_not_necessary_for_p_l1066_106680


namespace NUMINAMATH_CALUDE_unit_digit_of_8_power_1533_l1066_106696

theorem unit_digit_of_8_power_1533 : (8^1533 : ℕ) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_8_power_1533_l1066_106696


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l1066_106609

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 3 * x - 4 * y = -7 ∧ 6 * x - 5 * y = 5 :=
by
  use 7, 7
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l1066_106609


namespace NUMINAMATH_CALUDE_smallest_odd_four_primes_l1066_106655

def is_prime_factor (p n : ℕ) : Prop := Nat.Prime p ∧ p ∣ n

theorem smallest_odd_four_primes : 
  ∀ n : ℕ, 
    n % 2 = 1 → 
    (∃ p₁ p₂ p₃ p₄ : ℕ, 
      p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
      3 < p₁ ∧
      is_prime_factor p₁ n ∧
      is_prime_factor p₂ n ∧
      is_prime_factor p₃ n ∧
      is_prime_factor p₄ n ∧
      n = p₁ * p₂ * p₃ * p₄) →
    5005 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_four_primes_l1066_106655


namespace NUMINAMATH_CALUDE_probability_of_mixed_selection_l1066_106694

theorem probability_of_mixed_selection (n_boys n_girls n_select : ℕ) :
  n_boys = 5 →
  n_girls = 2 →
  n_select = 3 →
  (Nat.choose (n_boys + n_girls) n_select - Nat.choose n_boys n_select - Nat.choose n_girls n_select) / Nat.choose (n_boys + n_girls) n_select = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_mixed_selection_l1066_106694


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l1066_106670

theorem last_two_digits_sum (n : ℕ) : (9^n + 11^n) % 100 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l1066_106670


namespace NUMINAMATH_CALUDE_heather_final_blocks_l1066_106614

-- Define the initial number of blocks Heather has
def heather_initial : ℝ := 86.0

-- Define the number of blocks Jose shares
def jose_shares : ℝ := 41.0

-- Theorem statement
theorem heather_final_blocks : 
  heather_initial + jose_shares = 127.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_final_blocks_l1066_106614


namespace NUMINAMATH_CALUDE_uncle_dave_nieces_l1066_106638

theorem uncle_dave_nieces (ice_cream_per_niece : ℝ) (total_ice_cream : ℕ) 
  (h1 : ice_cream_per_niece = 143.0)
  (h2 : total_ice_cream = 1573) :
  (total_ice_cream : ℝ) / ice_cream_per_niece = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_nieces_l1066_106638


namespace NUMINAMATH_CALUDE_plane_line_relationship_l1066_106623

-- Define the types for planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (a : Set (ℝ × ℝ × ℝ))

-- Define the perpendicular relation
def perpendicular (S T : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the parallel relation
def parallel (S T : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the contained relation
def contained (L P : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem plane_line_relationship 
  (h1 : perpendicular α β) 
  (h2 : perpendicular a β) : 
  contained a α ∨ parallel a α := by sorry

end NUMINAMATH_CALUDE_plane_line_relationship_l1066_106623


namespace NUMINAMATH_CALUDE_fraction_inequality_triangle_sine_inequality_l1066_106629

-- Part 1
theorem fraction_inequality (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hmn : m > n) :
  n / m < (n + p) / (m + p) := by sorry

-- Part 2
theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (Real.sin C) / (Real.sin A + Real.sin B) + 
  (Real.sin A) / (Real.sin B + Real.sin C) + 
  (Real.sin B) / (Real.sin C + Real.sin A) < 2 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_triangle_sine_inequality_l1066_106629


namespace NUMINAMATH_CALUDE_empty_truck_weight_l1066_106667

-- Define the constants
def bridge_limit : ℕ := 20000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000
def loaded_truck_weight : ℕ := 24000

-- Define the theorem
theorem empty_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let dryers_weight := dryers * dryer_weight
  let cargo_weight := soda_weight + produce_weight + dryers_weight
  loaded_truck_weight - cargo_weight = 12000 := by
  sorry


end NUMINAMATH_CALUDE_empty_truck_weight_l1066_106667


namespace NUMINAMATH_CALUDE_stamp_denominations_l1066_106637

/-- Given stamps of denominations 7, n, and n+2 cents, 
    if 120 cents is the greatest postage that cannot be formed, then n = 22 -/
theorem stamp_denominations (n : ℕ) : 
  (∀ k > 120, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 2) * c) ∧
  (¬ ∃ a b c : ℕ, 120 = 7 * a + n * b + (n + 2) * c) →
  n = 22 := by sorry

end NUMINAMATH_CALUDE_stamp_denominations_l1066_106637


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l1066_106646

theorem max_value_x_plus_y (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 6 * y) :
  ∃ (max : ℝ), ∀ (x' y' : ℝ), 2 * x'^2 + 3 * y'^2 = 6 * y' → x' + y' ≤ max ∧ max = 1 + Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l1066_106646


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1066_106656

theorem inscribed_circle_radius (r : ℝ) :
  r > 0 →
  ∃ (R : ℝ), R > 0 ∧ R = 4 →
  r + r * Real.sqrt 2 = R →
  r = 4 * Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1066_106656


namespace NUMINAMATH_CALUDE_carol_final_gold_tokens_l1066_106695

/-- Represents the state of Carol's tokens -/
structure TokenState where
  purple : ℕ
  green : ℕ
  gold : ℕ

/-- Defines the exchange rules -/
def exchange1 (state : TokenState) : TokenState :=
  { purple := state.purple - 3, green := state.green + 2, gold := state.gold + 1 }

def exchange2 (state : TokenState) : TokenState :=
  { purple := state.purple + 1, green := state.green - 4, gold := state.gold + 1 }

/-- Checks if an exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.purple ≥ 3 ∨ state.green ≥ 4

/-- The initial state of Carol's tokens -/
def initialState : TokenState :=
  { purple := 100, green := 85, gold := 0 }

/-- The theorem to prove -/
theorem carol_final_gold_tokens :
  ∃ (finalState : TokenState),
    (¬canExchange finalState) ∧
    (finalState.gold = 90) ∧
    (∃ (n m : ℕ),
      finalState = (exchange2^[m] ∘ exchange1^[n]) initialState) :=
sorry

end NUMINAMATH_CALUDE_carol_final_gold_tokens_l1066_106695


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1066_106674

/-- An arithmetic sequence and its partial sums with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2
  S5_lt_S6 : S 5 < S 6
  S6_eq_S7 : S 6 = S 7
  S7_gt_S8 : S 7 > S 8

/-- The common difference of the arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 6 > 0 ∧ seq.a 7 = 0 ∧ seq.a 8 < 0 ∧ common_difference seq < 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1066_106674


namespace NUMINAMATH_CALUDE_library_fee_calculation_l1066_106602

/-- Calculates the total amount paid for borrowing books from a library. -/
def calculate_library_fee (daily_rate : ℚ) (book1_days : ℕ) (book2_days : ℕ) (book3_days : ℕ) : ℚ :=
  daily_rate * (book1_days + book2_days + book3_days)

theorem library_fee_calculation :
  let daily_rate : ℚ := 1/2
  let book1_days : ℕ := 20
  let book2_days : ℕ := 31
  let book3_days : ℕ := 31
  calculate_library_fee daily_rate book1_days book2_days book3_days = 41 := by
  sorry

#eval calculate_library_fee (1/2) 20 31 31

end NUMINAMATH_CALUDE_library_fee_calculation_l1066_106602


namespace NUMINAMATH_CALUDE_average_gas_mileage_round_trip_l1066_106628

/-- Calculates the average gas mileage for a round trip with different distances and fuel efficiencies -/
theorem average_gas_mileage_round_trip 
  (distance_outgoing : ℝ) 
  (distance_return : ℝ)
  (efficiency_outgoing : ℝ)
  (efficiency_return : ℝ) :
  let total_distance := distance_outgoing + distance_return
  let total_fuel := distance_outgoing / efficiency_outgoing + distance_return / efficiency_return
  let average_mileage := total_distance / total_fuel
  (distance_outgoing = 150 ∧ 
   distance_return = 180 ∧ 
   efficiency_outgoing = 25 ∧ 
   efficiency_return = 50) →
  (34 < average_mileage ∧ average_mileage < 35) :=
by sorry

end NUMINAMATH_CALUDE_average_gas_mileage_round_trip_l1066_106628


namespace NUMINAMATH_CALUDE_gcd_of_four_numbers_l1066_106673

theorem gcd_of_four_numbers : Nat.gcd 84 (Nat.gcd 108 (Nat.gcd 132 156)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_four_numbers_l1066_106673


namespace NUMINAMATH_CALUDE_compound_statement_falsity_l1066_106679

theorem compound_statement_falsity (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_compound_statement_falsity_l1066_106679


namespace NUMINAMATH_CALUDE_construction_cost_equation_l1066_106681

/-- The cost of land per square meter that satisfies the construction cost equation -/
def land_cost_per_sqm : ℝ := 50

/-- The cost of bricks per 1000 bricks -/
def brick_cost_per_1000 : ℝ := 100

/-- The cost of roof tiles per tile -/
def roof_tile_cost : ℝ := 10

/-- The required land area in square meters -/
def required_land_area : ℝ := 2000

/-- The required number of bricks -/
def required_bricks : ℝ := 10000

/-- The required number of roof tiles -/
def required_roof_tiles : ℝ := 500

/-- The total construction cost -/
def total_construction_cost : ℝ := 106000

theorem construction_cost_equation :
  land_cost_per_sqm * required_land_area +
  brick_cost_per_1000 * (required_bricks / 1000) +
  roof_tile_cost * required_roof_tiles =
  total_construction_cost :=
sorry

end NUMINAMATH_CALUDE_construction_cost_equation_l1066_106681


namespace NUMINAMATH_CALUDE_expression_evaluation_l1066_106669

theorem expression_evaluation : 16^3 + 3*(16^2) + 3*16 + 1 = 4913 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1066_106669


namespace NUMINAMATH_CALUDE_decimal_point_shift_l1066_106611

theorem decimal_point_shift (x : ℝ) (h : x > 0) :
  1000 * x = 3 * (1 / x) → x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l1066_106611


namespace NUMINAMATH_CALUDE_min_f_1998_l1066_106635

theorem min_f_1998 (f : ℕ → ℕ) 
  (h : ∀ s t : ℕ, f (t^2 * f s) = s * (f t)^2) : 
  f 1998 ≥ 1998 := by
  sorry

end NUMINAMATH_CALUDE_min_f_1998_l1066_106635


namespace NUMINAMATH_CALUDE_sum_mod_five_zero_l1066_106660

theorem sum_mod_five_zero : (4283 + 4284 + 4285 + 4286 + 4287) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_five_zero_l1066_106660


namespace NUMINAMATH_CALUDE_mother_age_twice_lisa_l1066_106634

/-- Lisa's birth year -/
def lisa_birth_year : ℕ := 1994

/-- The year of Lisa's 10th birthday -/
def reference_year : ℕ := 2004

/-- Lisa's age in the reference year -/
def lisa_age_reference : ℕ := 10

/-- Lisa's mother's age multiplier in the reference year -/
def mother_age_multiplier_reference : ℕ := 5

/-- The year when Lisa's mother's age is twice Lisa's age -/
def target_year : ℕ := 2034

theorem mother_age_twice_lisa (y : ℕ) :
  (y - lisa_birth_year) * 2 = (y - lisa_birth_year + mother_age_multiplier_reference * lisa_age_reference - lisa_age_reference) →
  y = target_year :=
by sorry

end NUMINAMATH_CALUDE_mother_age_twice_lisa_l1066_106634


namespace NUMINAMATH_CALUDE_derivative_at_pi_over_four_l1066_106621

theorem derivative_at_pi_over_four (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x * (Real.cos x + 1)) :
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_over_four_l1066_106621


namespace NUMINAMATH_CALUDE_largest_multiple_of_5_and_6_under_1000_l1066_106641

theorem largest_multiple_of_5_and_6_under_1000 :
  ∃ n : ℕ, n = 990 ∧
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_5_and_6_under_1000_l1066_106641


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l1066_106659

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the number of students to be selected -/
def num_selected : ℕ := 2

/-- Represents the event "at least 1 girl" -/
def at_least_one_girl : Set (Fin num_boys × Fin num_girls) := sorry

/-- Represents the event "all boys" -/
def all_boys : Set (Fin num_boys × Fin num_girls) := sorry

/-- Proves that the events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  (at_least_one_girl ∩ all_boys = ∅) ∧
  (at_least_one_girl ∪ all_boys = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l1066_106659


namespace NUMINAMATH_CALUDE_min_tangent_length_l1066_106648

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define point A
def point_A : ℝ × ℝ := (-1, 1)

-- Define the property that P is outside C
def outside_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 > 0

-- Define the tangent condition (|PM| = |PA|)
def tangent_condition (x y : ℝ) : Prop :=
  ∃ (mx my : ℝ), circle_C mx my ∧
  (x - mx)^2 + (y - my)^2 = (x + 1)^2 + (y - 1)^2

-- Theorem statement
theorem min_tangent_length :
  ∃ (min_length : ℝ),
    (∀ (x y : ℝ), outside_circle x y → tangent_condition x y →
      (x + 1)^2 + (y - 1)^2 ≥ min_length^2) ∧
    (∃ (x y : ℝ), outside_circle x y ∧ tangent_condition x y ∧
      (x + 1)^2 + (y - 1)^2 = min_length^2) ∧
    min_length = 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_tangent_length_l1066_106648


namespace NUMINAMATH_CALUDE_expected_digits_is_1_55_l1066_106620

/-- A fair 20-sided die with numbers from 1 to 20 -/
def icosahedral_die : Finset ℕ := Finset.range 20

/-- The probability of rolling any specific number on the die -/
def prob_roll (n : ℕ) : ℚ := if n ∈ icosahedral_die then 1 / 20 else 0

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := 
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the icosahedral die -/
def expected_digits : ℚ := 
  (icosahedral_die.sum (λ n => prob_roll n * num_digits n))

/-- Theorem: The expected number of digits when rolling a fair 20-sided die 
    with numbers from 1 to 20 is 1.55 -/
theorem expected_digits_is_1_55 : expected_digits = 31 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_1_55_l1066_106620


namespace NUMINAMATH_CALUDE_original_lettuce_price_l1066_106676

/-- Grocery order with item substitutions -/
def grocery_order (original_total delivery_tip new_total original_tomatoes new_tomatoes
                   original_celery new_celery new_lettuce : ℚ) : Prop :=
  -- Original order total before changes
  original_total = 25 ∧
  -- Delivery and tip
  delivery_tip = 8 ∧
  -- New total after changes and delivery/tip
  new_total = 35 ∧
  -- Original and new prices for tomatoes
  original_tomatoes = 0.99 ∧
  new_tomatoes = 2.20 ∧
  -- Original and new prices for celery
  original_celery = 1.96 ∧
  new_celery = 2 ∧
  -- New price for lettuce
  new_lettuce = 1.75

/-- The cost of the original lettuce -/
def original_lettuce_cost (original_total delivery_tip new_total original_tomatoes new_tomatoes
                           original_celery new_celery new_lettuce : ℚ) : ℚ :=
  new_lettuce - ((new_total - delivery_tip) - (original_total + (new_tomatoes - original_tomatoes) + (new_celery - original_celery)))

theorem original_lettuce_price
  (original_total delivery_tip new_total original_tomatoes new_tomatoes
   original_celery new_celery new_lettuce : ℚ)
  (h : grocery_order original_total delivery_tip new_total original_tomatoes new_tomatoes
                     original_celery new_celery new_lettuce) :
  original_lettuce_cost original_total delivery_tip new_total original_tomatoes new_tomatoes
                        original_celery new_celery new_lettuce = 1 := by
  sorry

end NUMINAMATH_CALUDE_original_lettuce_price_l1066_106676


namespace NUMINAMATH_CALUDE_stamp_redistribution_l1066_106616

/-- Represents the stamp redistribution problem -/
theorem stamp_redistribution (
  initial_albums : ℕ)
  (initial_pages_per_album : ℕ)
  (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ)
  (filled_albums : ℕ)
  (h1 : initial_albums = 10)
  (h2 : initial_pages_per_album = 50)
  (h3 : initial_stamps_per_page = 7)
  (h4 : new_stamps_per_page = 12)
  (h5 : filled_albums = 6) :
  (initial_albums * initial_pages_per_album * initial_stamps_per_page) % new_stamps_per_page = 8 := by
  sorry

#check stamp_redistribution

end NUMINAMATH_CALUDE_stamp_redistribution_l1066_106616


namespace NUMINAMATH_CALUDE_train_travel_time_l1066_106672

/-- Proves that a train traveling at 120 kmph for 80 km takes 40 minutes -/
theorem train_travel_time (speed : ℝ) (distance : ℝ) (time : ℝ) :
  speed = 120 →
  distance = 80 →
  time = distance / speed * 60 →
  time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l1066_106672


namespace NUMINAMATH_CALUDE_abs_h_eq_half_l1066_106665

/-- Given a quadratic equation x^2 - 4hx = 8, if the sum of squares of its roots is 20,
    then the absolute value of h is 1/2 -/
theorem abs_h_eq_half (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 8 ∧ y^2 - 4*h*y = 8 ∧ x^2 + y^2 = 20) → |h| = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_h_eq_half_l1066_106665


namespace NUMINAMATH_CALUDE_vector_dot_product_roots_l1066_106650

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.sqrt 3 * (Real.cos x)^2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, -2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_roots (x₁ x₂ : ℝ) :
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π ∧
  dot_product (m x₁) (n x₁) = 1/2 - Real.sqrt 3 ∧
  dot_product (m x₂) (n x₂) = 1/2 - Real.sqrt 3 →
  Real.sin (x₁ - x₂) = -Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_roots_l1066_106650


namespace NUMINAMATH_CALUDE_complex_division_l1066_106610

theorem complex_division (z₁ z₂ : ℂ) : z₁ = 1 + I ∧ z₂ = 2 * I → z₂ / z₁ = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l1066_106610


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l1066_106622

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 3 * a + b = a^2 + a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3 * x + y = x^2 + x * y → 2 * x + y ≥ 2 * Real.sqrt 2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l1066_106622


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1066_106606

/-- The number of ways to select a team from a math club -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (experienced_boys : ℕ) 
                (team_size : ℕ) (required_exp_boys : ℕ) (required_boys : ℕ) (required_girls : ℕ) : ℕ :=
  Nat.choose experienced_boys required_exp_boys * 
  Nat.choose (total_boys - experienced_boys) (required_boys - required_exp_boys) * 
  Nat.choose total_girls required_girls

theorem math_club_team_selection :
  select_team 9 10 4 7 2 4 3 = 7200 :=
by sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1066_106606


namespace NUMINAMATH_CALUDE_max_subsets_l1066_106619

/-- A set with 10 elements -/
def T : Finset (Fin 10) := Finset.univ

/-- The type of 5-element subsets of T -/
def Subset5 : Type := {S : Finset (Fin 10) // S.card = 5}

/-- The property that any two elements appear together in at most two subsets -/
def AtMostTwice (subsets : List Subset5) : Prop :=
  ∀ x y : Fin 10, x ≠ y → (subsets.filter (λ S => x ∈ S.1 ∧ y ∈ S.1)).length ≤ 2

/-- The main theorem -/
theorem max_subsets :
  (∃ subsets : List Subset5, AtMostTwice subsets ∧ subsets.length = 8) ∧
  (∀ subsets : List Subset5, AtMostTwice subsets → subsets.length ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_l1066_106619


namespace NUMINAMATH_CALUDE_expression_evaluation_l1066_106653

theorem expression_evaluation :
  let a : ℤ := 1001
  let b : ℤ := 1002
  let c : ℤ := 1000
  b^3 - a*b^2 - a^2*b + a^3 - c^3 = 2009007 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1066_106653


namespace NUMINAMATH_CALUDE_compute_expression_l1066_106671

theorem compute_expression : 3 * 3^4 - 27^65 / 27^63 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1066_106671


namespace NUMINAMATH_CALUDE_angle_sine_relation_l1066_106683

open Real

/-- For angles α and β in the first quadrant, "α > β" is neither a sufficient nor a necessary condition for "sin α > sin β". -/
theorem angle_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  (∃ α' β' : ℝ, α' > β' ∧ sin α' = sin β') ∧
  (∃ α'' β'' : ℝ, sin α'' > sin β'' ∧ ¬(α'' > β'')) := by
  sorry


end NUMINAMATH_CALUDE_angle_sine_relation_l1066_106683


namespace NUMINAMATH_CALUDE_congruence_problem_l1066_106666

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 19 = 3 → (3 * x + 14) % 19 = 18 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1066_106666


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l1066_106645

theorem sign_sum_theorem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let sign (x : ℝ) := x / |x|
  let expr := sign p + sign q + sign r + sign (p * q * r) + sign (p * q)
  expr = 5 ∨ expr = 1 ∨ expr = -1 :=
by sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l1066_106645


namespace NUMINAMATH_CALUDE_mean_median_difference_l1066_106625

/-- Represents the frequency histogram data for student absences -/
structure AbsenceData where
  zero_days : Nat
  one_day : Nat
  two_days : Nat
  three_days : Nat
  four_days : Nat
  total_students : Nat
  sum_condition : zero_days + one_day + two_days + three_days + four_days = total_students

/-- Calculates the mean number of days absent -/
def calculate_mean (data : AbsenceData) : Rat :=
  (0 * data.zero_days + 1 * data.one_day + 2 * data.two_days + 3 * data.three_days + 4 * data.four_days) / data.total_students

/-- Calculates the median number of days absent -/
def calculate_median (data : AbsenceData) : Nat :=
  if data.zero_days + data.one_day < data.total_students / 2 then 2 else 1

/-- Theorem stating the difference between mean and median -/
theorem mean_median_difference (data : AbsenceData) 
  (h : data.total_students = 20 ∧ 
       data.zero_days = 4 ∧ 
       data.one_day = 2 ∧ 
       data.two_days = 5 ∧ 
       data.three_days = 6 ∧ 
       data.four_days = 3) : 
  calculate_mean data - calculate_median data = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1066_106625


namespace NUMINAMATH_CALUDE_f_sum_equals_two_l1066_106649

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem f_sum_equals_two : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_two_l1066_106649


namespace NUMINAMATH_CALUDE_coin_flip_problem_l1066_106613

theorem coin_flip_problem (p_heads : ℝ) (p_event : ℝ) (n : ℕ) :
  p_heads = 1/2 →
  p_event = 0.03125 →
  p_event = p_heads * (1 - p_heads)^4 →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l1066_106613


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_is_six_l1066_106662

theorem sum_of_w_and_y_is_six (W X Y Z : ℕ) : 
  W ∈ ({1, 2, 3, 5} : Set ℕ) → 
  X ∈ ({1, 2, 3, 5} : Set ℕ) → 
  Y ∈ ({1, 2, 3, 5} : Set ℕ) → 
  Z ∈ ({1, 2, 3, 5} : Set ℕ) → 
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / X + (Y : ℚ) / Z = 3 →
  W + Y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_is_six_l1066_106662


namespace NUMINAMATH_CALUDE_arrangement_count_l1066_106690

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of teachers per group -/
def teachers_per_group : ℕ := 1

/-- The number of students per group -/
def students_per_group : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := choose num_teachers teachers_per_group * choose num_students students_per_group

theorem arrangement_count :
  total_arrangements = 12 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l1066_106690


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l1066_106639

/-- Given that Bryan has 42 books in total and each bookshelf contains 2 books,
    prove that the number of bookshelves he has is 21. -/
theorem bryans_bookshelves :
  let total_books : ℕ := 42
  let books_per_shelf : ℕ := 2
  let num_shelves : ℕ := total_books / books_per_shelf
  num_shelves = 21 := by
  sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l1066_106639


namespace NUMINAMATH_CALUDE_expression_equality_l1066_106699

theorem expression_equality (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 1) / x) * ((y^2 - 1) / y) - ((x^2 - 1) / y) * ((y^2 - 1) / x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1066_106699


namespace NUMINAMATH_CALUDE_prize_distribution_l1066_106652

theorem prize_distribution (total_prize : ℕ) (num_prizes : ℕ) (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ)
  (h_total : total_prize = 4200)
  (h_num : num_prizes = 7)
  (h_first : first_prize = 800)
  (h_second : second_prize = 700)
  (h_third : third_prize = 300) :
  ∃ (x y z : ℕ),
    x + y + z = num_prizes ∧
    x * first_prize + y * second_prize + z * third_prize = total_prize ∧
    x = 1 ∧ y = 4 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_prize_distribution_l1066_106652


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l1066_106607

theorem square_minus_product_plus_square : 7^2 - 4*5 + 2^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l1066_106607


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1066_106685

/-- Given a line segment CD where C(6,-1) is one endpoint and M(4,3) is the midpoint,
    the product of the coordinates of point D is 14. -/
theorem midpoint_coordinate_product : 
  let C : ℝ × ℝ := (6, -1)
  let M : ℝ × ℝ := (4, 3)
  let D : ℝ × ℝ := (2 * M.1 - C.1, 2 * M.2 - C.2)  -- Midpoint formula solved for D
  (D.1 * D.2 = 14) := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1066_106685


namespace NUMINAMATH_CALUDE_max_sum_after_adding_pyramid_to_pentagonal_prism_l1066_106687

/-- Represents a three-dimensional geometric structure --/
structure GeometricStructure where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- A pentagonal prism --/
def pentagonalPrism : GeometricStructure :=
  { faces := 7, edges := 15, vertices := 10 }

/-- Adds a pyramid to a pentagonal face of the structure --/
def addPyramidToPentagonalFace (s : GeometricStructure) : GeometricStructure :=
  { faces := s.faces - 1 + 5,
    edges := s.edges + 5,
    vertices := s.vertices + 1 }

/-- Adds a pyramid to a rectangular face of the structure --/
def addPyramidToRectangularFace (s : GeometricStructure) : GeometricStructure :=
  { faces := s.faces - 1 + 4,
    edges := s.edges + 4,
    vertices := s.vertices + 1 }

/-- Calculates the sum of faces, edges, and vertices of a structure --/
def sumElements (s : GeometricStructure) : ℕ :=
  s.faces + s.edges + s.vertices

/-- Theorem: The maximum sum of exterior faces, vertices, and edges after adding a pyramid to a pentagonal prism is 42 --/
theorem max_sum_after_adding_pyramid_to_pentagonal_prism :
  (sumElements (addPyramidToPentagonalFace pentagonalPrism)) = 42 ∧
  ∀ s : GeometricStructure, 
    (s = addPyramidToPentagonalFace pentagonalPrism ∨ 
     s = addPyramidToRectangularFace pentagonalPrism) →
    sumElements s ≤ 42 :=
by sorry


end NUMINAMATH_CALUDE_max_sum_after_adding_pyramid_to_pentagonal_prism_l1066_106687


namespace NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l1066_106604

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- The probability of a point not being above the x-axis in a given parallelogram -/
def probabilityNotAboveXAxis (para : Parallelogram) : ℝ := sorry

/-- The specific parallelogram PQRS from the problem -/
def pqrs : Parallelogram :=
  { p := { x := 4, y := 4 }
  , q := { x := -2, y := -2 }
  , r := { x := -8, y := -2 }
  , s := { x := -2, y := 4 }
  }

theorem probability_not_above_x_axis_is_half :
  probabilityNotAboveXAxis pqrs = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l1066_106604


namespace NUMINAMATH_CALUDE_chaz_final_floor_l1066_106632

def elevator_problem (start_floor : ℕ) (first_down : ℕ) (second_down : ℕ) : ℕ :=
  start_floor - first_down - second_down

theorem chaz_final_floor :
  elevator_problem 11 2 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chaz_final_floor_l1066_106632


namespace NUMINAMATH_CALUDE_sum_of_roots_bounds_l1066_106615

theorem sum_of_roots_bounds (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 10) : 
  Real.sqrt 6 + Real.sqrt 2 ≤ Real.sqrt (6 - x^2) + Real.sqrt (6 - y^2) + Real.sqrt (6 - z^2) 
  ∧ Real.sqrt (6 - x^2) + Real.sqrt (6 - y^2) + Real.sqrt (6 - z^2) ≤ 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_bounds_l1066_106615


namespace NUMINAMATH_CALUDE_complex_subtraction_l1066_106608

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1066_106608


namespace NUMINAMATH_CALUDE_jumping_contest_l1066_106640

theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 14 →
  frog_jump = grasshopper_jump + 37 →
  mouse_jump = frog_jump - 16 →
  mouse_jump - grasshopper_jump = 21 :=
by sorry

end NUMINAMATH_CALUDE_jumping_contest_l1066_106640


namespace NUMINAMATH_CALUDE_frank_reading_rate_l1066_106617

/-- Given a book with a certain number of chapters read over a certain number of days,
    calculate the number of chapters read per day. -/
def chapters_per_day (total_chapters : ℕ) (days : ℕ) : ℚ :=
  (total_chapters : ℚ) / (days : ℚ)

/-- Theorem: For a book with 2 chapters read over 664 days,
    the number of chapters read per day is 2/664. -/
theorem frank_reading_rate : chapters_per_day 2 664 = 2 / 664 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_rate_l1066_106617


namespace NUMINAMATH_CALUDE_discount_rate_proof_l1066_106698

theorem discount_rate_proof (initial_price final_price : ℝ) 
  (h1 : initial_price = 7200)
  (h2 : final_price = 3528)
  (h3 : ∃ x : ℝ, initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.3 ∧ initial_price * (1 - x)^2 = final_price := by
sorry

end NUMINAMATH_CALUDE_discount_rate_proof_l1066_106698


namespace NUMINAMATH_CALUDE_id_number_2520_l1066_106605

/-- A type representing a 7-digit identification number -/
def IdNumber := Fin 7 → Fin 7

/-- The set of all valid identification numbers -/
def ValidIdNumbers : Set IdNumber :=
  {id | Function.Injective id ∧ (∀ i, id i < 7)}

/-- Lexicographical order on identification numbers -/
def IdLexOrder (id1 id2 : IdNumber) : Prop :=
  ∃ k, (∀ i < k, id1 i = id2 i) ∧ id1 k < id2 k

/-- The nth identification number in lexicographical order -/
noncomputable def nthIdNumber (n : ℕ) : IdNumber :=
  sorry

/-- The main theorem: the 2520th identification number is 4376521 -/
theorem id_number_2520 :
  nthIdNumber 2520 = λ i =>
    match i with
    | 0 => 3  -- 4 (0-indexed)
    | 1 => 5  -- 6
    | 2 => 2  -- 3
    | 3 => 5  -- 6
    | 4 => 4  -- 5
    | 5 => 1  -- 2
    | 6 => 0  -- 1
  := by sorry

end NUMINAMATH_CALUDE_id_number_2520_l1066_106605


namespace NUMINAMATH_CALUDE_largest_x_floor_fraction_l1066_106636

theorem largest_x_floor_fraction : 
  (∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) / x = 7 / 8) ∧ 
  (∀ (y : ℝ), y > 0 → (⌊y⌋ : ℝ) / y = 7 / 8 → y ≤ 48 / 7) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_fraction_l1066_106636


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1066_106633

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 5

/-- Whether red ribbon is available -/
def red_ribbon_available : Prop := true

/-- The number of invalid combinations due to supply issue -/
def invalid_combinations : ℕ := 5

theorem gift_wrapping_combinations : 
  wrapping_paper_varieties * ribbon_colors * gift_card_types - invalid_combinations = 195 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1066_106633


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1066_106644

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1066_106644


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1066_106689

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := 3 * (x + 1)^2 - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 6 * (x + 1)

theorem quadratic_function_properties :
  (f 1 = 10) ∧
  (f (-1) = -2) ∧
  (∀ x > -1, f' x > 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1066_106689


namespace NUMINAMATH_CALUDE_sams_remaining_dimes_l1066_106686

/-- Given Sam's initial number of dimes and the number of dimes borrowed by his sister,
    prove that the number of dimes Sam has now is equal to their difference. -/
theorem sams_remaining_dimes (initial_dimes borrowed_dimes : ℕ) :
  initial_dimes = 8 →
  borrowed_dimes = 4 →
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_dimes_l1066_106686


namespace NUMINAMATH_CALUDE_segment_length_l1066_106692

/-- Given two points P and Q on a line segment AB, prove that AB has length 336/11 -/
theorem segment_length (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are on AB and on the same side of midpoint
  (P - A) / (B - P) = 3 / 4 →        -- P divides AB in ratio 3:4
  (Q - A) / (B - Q) = 5 / 7 →        -- Q divides AB in ratio 5:7
  Q - P = 4 →                        -- PQ = 4
  B - A = 336 / 11 := by             -- AB has length 336/11
sorry


end NUMINAMATH_CALUDE_segment_length_l1066_106692


namespace NUMINAMATH_CALUDE_counting_sequence_53rd_term_l1066_106697

theorem counting_sequence_53rd_term : 
  let seq : ℕ → ℕ := λ n => n
  seq 53 = 10 := by
  sorry

end NUMINAMATH_CALUDE_counting_sequence_53rd_term_l1066_106697


namespace NUMINAMATH_CALUDE_roberts_reading_l1066_106657

/-- Given Robert's reading speed, book size, and available time, 
    prove the maximum number of complete books he can read. -/
theorem roberts_reading (
  reading_speed : ℕ) 
  (book_size : ℕ) 
  (available_time : ℕ) 
  (h1 : reading_speed = 120) 
  (h2 : book_size = 360) 
  (h3 : available_time = 8) : 
  (available_time * reading_speed) / book_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_roberts_reading_l1066_106657


namespace NUMINAMATH_CALUDE_combined_instruments_count_l1066_106642

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- Calculates the total number of instruments -/
def totalInstruments (i : Instruments) : ℕ :=
  i.flutes + i.horns + i.harps

/-- Charlie's instruments -/
def charlie : Instruments :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instruments -/
def carli : Instruments :=
  { flutes := 2 * charlie.flutes,
    horns := charlie.horns / 2,
    harps := 0 }

theorem combined_instruments_count :
  totalInstruments charlie + totalInstruments carli = 7 := by
  sorry

end NUMINAMATH_CALUDE_combined_instruments_count_l1066_106642


namespace NUMINAMATH_CALUDE_circle_radius_from_chord_and_secant_l1066_106654

/-- Given a circle with a chord of length 10 and a secant parallel to the tangent at one end of the chord,
    where the internal segment of the secant is 12 units long, the radius of the circle is 10 units. -/
theorem circle_radius_from_chord_and_secant (C : ℝ → ℝ → Prop) (A B M : ℝ × ℝ) (r : ℝ) :
  (∀ x y, C x y ↔ (x - r)^2 + (y - r)^2 = r^2) →  -- C is a circle with center (r, r) and radius r
  C A.1 A.2 →  -- A is on the circle
  C B.1 B.2 →  -- B is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100 →  -- AB is a chord of length 10
  (∃ t : ℝ, C (A.1 + t) (A.2 + t) ∧ (A.1 + t - B.1)^2 + (A.2 + t - B.2)^2 = 36) →  -- Secant parallel to tangent at A
  r = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_chord_and_secant_l1066_106654


namespace NUMINAMATH_CALUDE_oz_language_word_loss_l1066_106627

theorem oz_language_word_loss :
  let total_letters : ℕ := 64
  let forbidden_letters : ℕ := 1
  let max_word_length : ℕ := 2

  let one_letter_words_lost : ℕ := forbidden_letters
  let two_letter_words_lost : ℕ := 
    total_letters * forbidden_letters + 
    forbidden_letters * total_letters - 
    forbidden_letters * forbidden_letters

  one_letter_words_lost + two_letter_words_lost = 128 :=
by sorry

end NUMINAMATH_CALUDE_oz_language_word_loss_l1066_106627


namespace NUMINAMATH_CALUDE_D_72_l1066_106691

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order of factors matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- Theorem stating that D(72) = 35 -/
theorem D_72 : D 72 = 35 := by sorry

end NUMINAMATH_CALUDE_D_72_l1066_106691
