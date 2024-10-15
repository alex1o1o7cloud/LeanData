import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_x_l2355_235549

theorem solve_for_x (M N : ℝ) (h1 : M = 2*x - 4) (h2 : N = 2*x + 3) (h3 : 3*M - N = 1) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2355_235549


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l2355_235555

/-- For an arithmetic sequence {a_n} with first term 1 and common difference 3,
    prove that the 100th term is 298. -/
theorem arithmetic_sequence_100th_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n + 3) →  -- arithmetic sequence with common difference 3
  a 1 = 1 →                    -- first term is 1
  a 100 = 298 := by             -- 100th term is 298
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l2355_235555


namespace NUMINAMATH_CALUDE_jump_difference_l2355_235503

def monday_jumps : ℕ := 88
def tuesday_jumps : ℕ := 75
def wednesday_jumps : ℕ := 62
def thursday_jumps : ℕ := 91
def friday_jumps : ℕ := 80

def jump_counts : List ℕ := [monday_jumps, tuesday_jumps, wednesday_jumps, thursday_jumps, friday_jumps]

theorem jump_difference :
  (List.maximum jump_counts).get! - (List.minimum jump_counts).get! = 29 := by
  sorry

end NUMINAMATH_CALUDE_jump_difference_l2355_235503


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_equation_l2355_235543

theorem x_cubed_coefficient_equation (a : ℝ) : 
  (∃ k : ℝ, k = 56 ∧ k = 6 * a^2 - 15 * a + 20) ↔ (a = 6 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_equation_l2355_235543


namespace NUMINAMATH_CALUDE_total_pencils_l2355_235533

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l2355_235533


namespace NUMINAMATH_CALUDE_problem_statement_l2355_235530

theorem problem_statement (a b : ℝ) : 
  |a - 3| + (b + 4)^2 = 0 → (a + b)^2003 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2355_235530


namespace NUMINAMATH_CALUDE_sarahs_friends_ages_sum_l2355_235528

theorem sarahs_friends_ages_sum :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →  -- single-digit integers
    a ≠ b ∧ b ≠ c ∧ a ≠ c →     -- distinct
    a * b = 36 →                -- product of two ages is 36
    c ∣ 36 →                    -- third age is a factor of 36
    c ≠ a ∧ c ≠ b →             -- third age is not one of the first two
    a + b + c = 16 :=           -- sum of all three ages is 16
by sorry

end NUMINAMATH_CALUDE_sarahs_friends_ages_sum_l2355_235528


namespace NUMINAMATH_CALUDE_chairs_count_l2355_235517

/-- The number of chairs in the auditorium at Yunju's school -/
def total_chairs : ℕ := by sorry

/-- The auditorium is square-shaped -/
axiom is_square : total_chairs = (Nat.sqrt total_chairs) ^ 2

/-- Yunju's seat is 2nd from the front -/
axiom front_distance : 2 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 5th from the back -/
axiom back_distance : 5 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 3rd from the right -/
axiom right_distance : 3 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 4th from the left -/
axiom left_distance : 4 ≤ Nat.sqrt total_chairs

/-- The theorem to be proved -/
theorem chairs_count : total_chairs = 36 := by sorry

end NUMINAMATH_CALUDE_chairs_count_l2355_235517


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2355_235509

/-- Given a quadratic equation x^2 + 8x - 1 = 0, when written in the form (x + a)^2 = b, b equals 17 -/
theorem complete_square_quadratic : 
  ∃ a : ℝ, ∀ x : ℝ, (x^2 + 8*x - 1 = 0) ↔ ((x + a)^2 = 17) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2355_235509


namespace NUMINAMATH_CALUDE_train_speed_difference_l2355_235564

theorem train_speed_difference (v : ℝ) 
  (cattle_speed : ℝ) (head_start : ℝ) (diesel_time : ℝ) (total_distance : ℝ)
  (h1 : v < cattle_speed)
  (h2 : cattle_speed = 56)
  (h3 : head_start = 6)
  (h4 : diesel_time = 12)
  (h5 : total_distance = 1284)
  (h6 : cattle_speed * head_start + cattle_speed * diesel_time + v * diesel_time = total_distance) :
  cattle_speed - v = 33 := by
sorry

end NUMINAMATH_CALUDE_train_speed_difference_l2355_235564


namespace NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_four_l2355_235524

def probability_at_least_one_multiple_of_four : ℚ :=
  let total_numbers : ℕ := 60
  let multiples_of_four : ℕ := 15
  let prob_not_multiple : ℚ := (total_numbers - multiples_of_four) / total_numbers
  1 - prob_not_multiple ^ 2

theorem prob_at_least_one_multiple_of_four :
  probability_at_least_one_multiple_of_four = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_four_l2355_235524


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l2355_235500

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry

-- Assume P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the distances from P to the foci
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Assume the ratio of PF₁ to PF₂ is 2:1
axiom distance_ratio : PF₁ = 2 * PF₂

-- Define the area of the triangle
def triangle_area : ℝ := sorry

-- State the theorem
theorem ellipse_triangle_area : triangle_area = 4 := sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l2355_235500


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l2355_235552

/-- Represents the number of ways to distribute books between the library and checked-out status. -/
def book_distribution_ways (n₁ n₂ : ℕ) : ℕ :=
  (n₁ - 1) * (n₂ - 1)

/-- Theorem stating the number of ways to distribute books between the library and checked-out status. -/
theorem book_distribution_theorem :
  let n₁ : ℕ := 8  -- number of copies of the first type of book
  let n₂ : ℕ := 4  -- number of copies of the second type of book
  book_distribution_ways n₁ n₂ = 21 :=
by
  sorry

#eval book_distribution_ways 8 4

end NUMINAMATH_CALUDE_book_distribution_theorem_l2355_235552


namespace NUMINAMATH_CALUDE_impossibleArrangement_l2355_235541

/-- Represents a person at the table -/
structure Person :=
  (index : Fin 40)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Calculates the number of people between two given people -/
def distance (table : Table) (p1 p2 : Person) : Nat :=
  sorry

/-- Determines if two people have a mutual acquaintance -/
def hasCommonAcquaintance (table : Table) (p1 p2 : Person) : Prop :=
  sorry

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossibleArrangement :
  ¬ ∃ (table : Table),
    (∀ (p1 p2 : Person),
      hasCommonAcquaintance table p1 p2 ↔ Even (distance table p1 p2)) :=
  sorry

end NUMINAMATH_CALUDE_impossibleArrangement_l2355_235541


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2355_235511

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2355_235511


namespace NUMINAMATH_CALUDE_states_fraction_1840_to_1849_l2355_235568

theorem states_fraction_1840_to_1849 (total_states : ℕ) (joined_1840_to_1849 : ℕ) :
  total_states = 33 →
  joined_1840_to_1849 = 6 →
  (joined_1840_to_1849 : ℚ) / total_states = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_1840_to_1849_l2355_235568


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l2355_235590

def S : Set Int := {7, 25, -1, 12, -3}

theorem smallest_sum_of_three (s : Set Int) (h : s = S) :
  (∃ (a b c : Int), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a + b + c = 3 ∧
    ∀ (x y z : Int), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
      x + y + z ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l2355_235590


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2355_235526

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 6 }

/-- Theorem: The maximum number of soap boxes that can be placed in the carton is 250 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 250 := by
  sorry


end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2355_235526


namespace NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l2355_235581

theorem max_value_fraction (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable : 
  ∃ x y : ℝ, (2*x + 3*y + 4) / Real.sqrt (x^2 + y^2 + 2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l2355_235581


namespace NUMINAMATH_CALUDE_function_roots_l2355_235510

def has_at_least_roots (f : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card ≥ n ∧ (∀ x ∈ S, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem function_roots (g : ℝ → ℝ) 
  (h1 : ∀ x, g (3 + x) = g (3 - x))
  (h2 : ∀ x, g (8 + x) = g (8 - x))
  (h3 : g 0 = 0) :
  has_at_least_roots g 501 (-2000) 2000 := by
  sorry

end NUMINAMATH_CALUDE_function_roots_l2355_235510


namespace NUMINAMATH_CALUDE_lifeguard_swimming_distance_l2355_235504

/-- The problem of calculating the total swimming distance for a lifeguard test. -/
theorem lifeguard_swimming_distance 
  (front_crawl_speed : ℝ) 
  (breaststroke_speed : ℝ) 
  (total_time : ℝ) 
  (front_crawl_time : ℝ) 
  (h1 : front_crawl_speed = 45) 
  (h2 : breaststroke_speed = 35) 
  (h3 : total_time = 12) 
  (h4 : front_crawl_time = 8) :
  front_crawl_speed * front_crawl_time + breaststroke_speed * (total_time - front_crawl_time) = 500 := by
  sorry

#check lifeguard_swimming_distance

end NUMINAMATH_CALUDE_lifeguard_swimming_distance_l2355_235504


namespace NUMINAMATH_CALUDE_olives_price_per_pound_l2355_235589

/-- Calculates the price per pound of olives given Teresa's shopping list and total spent --/
theorem olives_price_per_pound (sandwich_price : ℝ) (salami_price : ℝ) (olive_weight : ℝ) 
  (feta_weight : ℝ) (feta_price_per_pound : ℝ) (bread_price : ℝ) (total_spent : ℝ) 
  (h1 : sandwich_price = 7.75)
  (h2 : salami_price = 4)
  (h3 : olive_weight = 1/4)
  (h4 : feta_weight = 1/2)
  (h5 : feta_price_per_pound = 8)
  (h6 : bread_price = 2)
  (h7 : total_spent = 40) :
  (total_spent - (2 * sandwich_price + salami_price + 3 * salami_price + 
  feta_weight * feta_price_per_pound + bread_price)) / olive_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_olives_price_per_pound_l2355_235589


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2355_235597

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (12^3 + 15^4 - 6^5) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (12^3 + 15^4 - 6^5) → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2355_235597


namespace NUMINAMATH_CALUDE_sweet_shop_candy_cases_l2355_235505

/-- The number of cases of chocolate bars in the Sweet Shop -/
def chocolate_cases : ℕ := 80 - 55

/-- The total number of cases of candy in the Sweet Shop -/
def total_cases : ℕ := 80

/-- The number of cases of lollipops in the Sweet Shop -/
def lollipop_cases : ℕ := 55

theorem sweet_shop_candy_cases : chocolate_cases = 25 := by
  sorry

end NUMINAMATH_CALUDE_sweet_shop_candy_cases_l2355_235505


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2355_235546

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2355_235546


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l2355_235512

theorem triangle_trig_identity (D E F : Real) (DE DF EF : Real) : 
  DE = 7 → DF = 8 → EF = 5 → 
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - 
  (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l2355_235512


namespace NUMINAMATH_CALUDE_order_relation_l2355_235547

theorem order_relation (a b c d : ℝ) 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l2355_235547


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l2355_235516

theorem sqrt_D_irrational (k : ℤ) : 
  let a : ℤ := 3 * k
  let b : ℤ := 3 * k + 3
  let c : ℤ := a + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt D) := by sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l2355_235516


namespace NUMINAMATH_CALUDE_irrational_between_neg_three_and_neg_two_l2355_235540

theorem irrational_between_neg_three_and_neg_two :
  ∃ x : ℝ, Irrational x ∧ -3 < x ∧ x < -2 := by sorry

end NUMINAMATH_CALUDE_irrational_between_neg_three_and_neg_two_l2355_235540


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2355_235544

theorem sum_with_radical_conjugate :
  let x : ℝ := 5 - Real.sqrt 500
  let y : ℝ := 5 + Real.sqrt 500
  x + y = 10 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2355_235544


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l2355_235548

theorem negation_of_forall_positive (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l2355_235548


namespace NUMINAMATH_CALUDE_mr_a_net_gain_l2355_235508

/-- The total net gain for Mr. A after a series of house transactions -/
theorem mr_a_net_gain (house1_value house2_value : ℝ)
  (house1_profit house1_loss house2_profit house2_loss : ℝ)
  (h1 : house1_value = 15000)
  (h2 : house2_value = 20000)
  (h3 : house1_profit = 0.15)
  (h4 : house1_loss = 0.15)
  (h5 : house2_profit = 0.20)
  (h6 : house2_loss = 0.20) :
  let sale1 := house1_value * (1 + house1_profit)
  let sale2 := house2_value * (1 + house2_profit)
  let buyback1 := sale1 * (1 - house1_loss)
  let buyback2 := sale2 * (1 - house2_loss)
  let net_gain := (sale1 - buyback1) + (sale2 - buyback2)
  net_gain = 7387.50 := by
  sorry

end NUMINAMATH_CALUDE_mr_a_net_gain_l2355_235508


namespace NUMINAMATH_CALUDE_total_houses_l2355_235553

theorem total_houses (dogs : ℕ) (cats : ℕ) (both : ℕ) (h1 : dogs = 40) (h2 : cats = 30) (h3 : both = 10) :
  dogs + cats - both = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_houses_l2355_235553


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2355_235551

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  para a b := by sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2355_235551


namespace NUMINAMATH_CALUDE_runner_speed_ratio_l2355_235593

theorem runner_speed_ratio (u₁ u₂ : ℝ) (h₁ : u₁ > u₂) (h₂ : u₁ + u₂ = 5) (h₃ : u₁ - u₂ = 5 / 3) :
  u₁ / u₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_runner_speed_ratio_l2355_235593


namespace NUMINAMATH_CALUDE_min_value_theorem_l2355_235570

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 4) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ (z : ℝ), (2/x + 1/y) ≥ z → z ≤ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2355_235570


namespace NUMINAMATH_CALUDE_congruent_triangles_sum_l2355_235520

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two triangles are congruent if their corresponding sides are equal -/
def congruent (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

theorem congruent_triangles_sum (x y : ℝ) :
  let t1 : Triangle := ⟨2, 5, x⟩
  let t2 : Triangle := ⟨y, 2, 6⟩
  congruent t1 t2 → x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_sum_l2355_235520


namespace NUMINAMATH_CALUDE_square_area_ratio_l2355_235562

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  (2 * s)^2 / s^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2355_235562


namespace NUMINAMATH_CALUDE_min_polyline_distance_circle_line_l2355_235569

/-- Polyline distance between two points -/
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- A point is on the unit circle -/
def on_unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- A point is on the given line -/
def on_line (x y : ℝ) : Prop :=
  2*x + y - 2*Real.sqrt 5 = 0

/-- The minimum polyline distance between the circle and the line -/
theorem min_polyline_distance_circle_line :
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), on_unit_circle x₁ y₁ → on_line x₂ y₂ →
      polyline_distance x₁ y₁ x₂ y₂ ≥ min_dist) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), on_unit_circle x₁ y₁ ∧ on_line x₂ y₂ ∧
      polyline_distance x₁ y₁ x₂ y₂ = min_dist) ∧
    min_dist = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_polyline_distance_circle_line_l2355_235569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2355_235550

/-- Given an arithmetic sequence where the third term is 23 and the sixth term is 29,
    prove that the ninth term is 35. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℕ)  -- a is the sequence
  (h1 : a 3 = 23)  -- third term is 23
  (h2 : a 6 = 29)  -- sixth term is 29
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence property
  : a 9 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2355_235550


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l2355_235537

/-- Represents the ages of a father and son -/
structure Ages where
  son : ℕ
  father : ℕ

/-- The current ages of the father and son -/
def currentAges : Ages :=
  { son := 24, father := 72 }

/-- The ages of the father and son 8 years ago -/
def pastAges : Ages :=
  { son := currentAges.son - 8, father := currentAges.father - 8 }

/-- The ratio of the father's age to the son's age -/
def ageRatio (ages : Ages) : ℚ :=
  ages.father / ages.son

theorem father_son_age_ratio :
  (pastAges.father = 4 * pastAges.son) →
  ageRatio currentAges = 3 / 1 := by
  sorry

#eval ageRatio currentAges

end NUMINAMATH_CALUDE_father_son_age_ratio_l2355_235537


namespace NUMINAMATH_CALUDE_sum_of_squares_l2355_235521

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 9)
  (eq2 : y^2 + 5*z = -9)
  (eq3 : z^2 + 7*x = -18) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2355_235521


namespace NUMINAMATH_CALUDE_toy_store_revenue_l2355_235518

theorem toy_store_revenue (december : ℝ) (november january : ℝ) 
  (h1 : november = (2/5) * december) 
  (h2 : january = (1/3) * november) : 
  december = 5 * ((november + january) / 2) := by
  sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l2355_235518


namespace NUMINAMATH_CALUDE_two_questions_suffice_l2355_235578

-- Define the possible types of siblings
inductive SiblingType
  | Truthful
  | Unpredictable

-- Define a sibling
structure Sibling :=
  (type : SiblingType)

-- Define the farm setup
structure Farm :=
  (siblings : Fin 3 → Sibling)
  (correct_path : Nat)

-- Define the possible answers to a question
inductive Answer
  | Yes
  | No

-- Define a question as a function from a sibling to an answer
def Question := Sibling → Answer

-- Define the theorem
theorem two_questions_suffice (farm : Farm) :
  ∃ (q1 q2 : Question), ∀ (i j : Fin 3),
    (farm.siblings i).type = SiblingType.Truthful →
    (farm.siblings j).type = SiblingType.Truthful →
    i ≠ j →
    ∃ (f : Answer → Answer → Nat),
      f (q1 (farm.siblings i)) (q2 (farm.siblings j)) = farm.correct_path :=
sorry


end NUMINAMATH_CALUDE_two_questions_suffice_l2355_235578


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l2355_235586

theorem smallest_number_with_remainder_two : ∃! n : ℕ,
  n > 1 ∧
  (∀ d ∈ ({3, 4, 5, 6, 7} : Set ℕ), n % d = 2) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ d ∈ ({3, 4, 5, 6, 7} : Set ℕ), m % d = 2) → m ≥ n) ∧
  n = 422 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l2355_235586


namespace NUMINAMATH_CALUDE_truck_travel_distance_l2355_235591

/-- Given a truck that travels 300 miles on 10 gallons of gas, 
    prove that it will travel 450 miles on 15 gallons of gas, 
    assuming a constant rate of fuel consumption. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_gas : ℝ) (new_gas : ℝ) : 
  initial_distance = 300 ∧ initial_gas = 10 ∧ new_gas = 15 →
  (new_gas * initial_distance) / initial_gas = 450 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l2355_235591


namespace NUMINAMATH_CALUDE_watch_time_calculation_l2355_235579

/-- The total watching time for two shows, where the second is 4 times longer than the first -/
def total_watching_time (first_show_duration : ℕ) : ℕ :=
  first_show_duration + 4 * first_show_duration

/-- Theorem stating that given a 30-minute show and another 4 times longer, the total watching time is 150 minutes -/
theorem watch_time_calculation : total_watching_time 30 = 150 := by
  sorry

end NUMINAMATH_CALUDE_watch_time_calculation_l2355_235579


namespace NUMINAMATH_CALUDE_bread_left_l2355_235577

theorem bread_left (total : ℕ) (bomi_ate : ℕ) (yejun_ate : ℕ) 
  (h1 : total = 1000)
  (h2 : bomi_ate = 350)
  (h3 : yejun_ate = 500) :
  total - (bomi_ate + yejun_ate) = 150 := by
  sorry

end NUMINAMATH_CALUDE_bread_left_l2355_235577


namespace NUMINAMATH_CALUDE_sin_30_cos_60_plus_cos_30_sin_60_l2355_235583

theorem sin_30_cos_60_plus_cos_30_sin_60 : 
  Real.sin (30 * π / 180) * Real.cos (60 * π / 180) + 
  Real.cos (30 * π / 180) * Real.sin (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_cos_60_plus_cos_30_sin_60_l2355_235583


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l2355_235587

-- Define the conversion factor
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def cubic_yards : ℝ := 5

-- Theorem to prove
theorem cubic_yards_to_cubic_feet :
  cubic_yards * yards_to_feet^3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l2355_235587


namespace NUMINAMATH_CALUDE_square_sum_factorial_solutions_l2355_235588

theorem square_sum_factorial_solutions :
  ∀ (a b n : ℕ+),
    n < 14 →
    a ≤ b →
    a ^ 2 + b ^ 2 = n! →
    ((n = 2 ∧ a = 1 ∧ b = 1) ∨ (n = 6 ∧ a = 12 ∧ b = 24)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_factorial_solutions_l2355_235588


namespace NUMINAMATH_CALUDE_function_value_negation_l2355_235514

/-- Given a function f(x) = a * sin(πx + α) + b * cos(πx + β) where f(2002) = 3,
    prove that f(2003) = -f(2002). -/
theorem function_value_negation (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2002 = 3 → f 2003 = -f 2002 := by
  sorry

end NUMINAMATH_CALUDE_function_value_negation_l2355_235514


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l2355_235574

/-- Given two vectors a and b in ℝ², where a = (2,1) and b = (x,2),
    if a + b is parallel to a - 2b, then x = 4 -/
theorem parallel_vectors_imply_x_equals_four (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 2)
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = k • (a.1 - 2*b.1, a.2 - 2*b.2)) →
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l2355_235574


namespace NUMINAMATH_CALUDE_curve_transformation_l2355_235529

theorem curve_transformation (x y : ℝ) : 
  y = Real.sin (π / 2 + 2 * x) → 
  y = -Real.cos (5 * π / 6 - 3 * ((2 / 3) * x - π / 18)) := by
sorry

end NUMINAMATH_CALUDE_curve_transformation_l2355_235529


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2355_235531

-- Problem 1
theorem simplify_expression_1 (x y z : ℝ) :
  (x + y + z)^2 - (x + y - z)^2 = 4*z*(x + y) := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  (a + 2*b)^2 - 2*(a + 2*b)*(a - 2*b) + (a - 2*b)^2 = 16*b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2355_235531


namespace NUMINAMATH_CALUDE_angle_approximation_l2355_235599

/-- Regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  center : ℝ × ℝ
  radius : ℝ
  vertices : Fin n → ℝ × ℝ

/-- Construct points B, C, D, E as described in the problem -/
def constructPoints (p : RegularPolygon 19) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Length of chord DE -/
def chordLength (p : RegularPolygon 19) : ℝ := sorry

/-- Angle formed by radii after 19 sequential measurements -/
def angleAfterMeasurements (p : RegularPolygon 19) : ℝ := sorry

/-- Main theorem: The angle formed after 19 measurements is approximately 4°57' -/
theorem angle_approximation (p : RegularPolygon 19) : 
  ∃ ε > 0, abs (angleAfterMeasurements p - (4 + 57/60) * π / 180) < ε :=
sorry

end NUMINAMATH_CALUDE_angle_approximation_l2355_235599


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_algebraic_simplification_l2355_235559

-- Part 1: Quadratic equation
theorem quadratic_equation_solution (x : ℝ) : 
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1/2 ∨ x = 1 := by sorry

-- Part 2: Algebraic simplification
theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  ((a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a)) / (b^2 / (a^2 - a*b)) = a / b := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_algebraic_simplification_l2355_235559


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2355_235515

/-- Given a street with parallel curbs and a crosswalk, calculate the distance between the stripes -/
theorem crosswalk_stripe_distance
  (curb_distance : ℝ)
  (curb_length : ℝ)
  (stripe_length : ℝ)
  (h_curb_distance : curb_distance = 50)
  (h_curb_length : curb_length = 20)
  (h_stripe_length : stripe_length = 65) :
  (curb_distance * curb_length) / stripe_length = 200 / 13 := by
sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2355_235515


namespace NUMINAMATH_CALUDE_man_speed_against_stream_l2355_235554

/-- Calculates the speed against the stream given the rate in still water and speed with the stream -/
def speed_against_stream (rate_still : ℝ) (speed_with_stream : ℝ) : ℝ :=
  |rate_still - (speed_with_stream - rate_still)|

/-- Theorem: Given a man's rate in still water of 2 km/h and speed with the stream of 6 km/h,
    his speed against the stream is 2 km/h -/
theorem man_speed_against_stream :
  speed_against_stream 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_stream_l2355_235554


namespace NUMINAMATH_CALUDE_translation_down_three_units_l2355_235560

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

theorem translation_down_three_units :
  let originalLine : Line := { slope := 1/2, intercept := 0 }
  let translatedLine : Line := translateLine originalLine 3
  translatedLine = { slope := 1/2, intercept := -3 } := by
  sorry

end NUMINAMATH_CALUDE_translation_down_three_units_l2355_235560


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l2355_235522

def baseball_cards (initial cards_from_fred cards_bought torn : ℕ) : ℕ :=
  initial - torn + cards_from_fred + cards_bought

theorem mary_baseball_cards : 
  baseball_cards 18 26 40 8 = 76 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l2355_235522


namespace NUMINAMATH_CALUDE_intersection_empty_iff_m_nonnegative_l2355_235566

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x + m = 0}
def B : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_empty_iff_m_nonnegative (m : ℝ) :
  A m ∩ B = ∅ ↔ m ∈ Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_m_nonnegative_l2355_235566


namespace NUMINAMATH_CALUDE_marys_stickers_l2355_235502

theorem marys_stickers (total_stickers : ℕ) (friends : ℕ) (other_students : ℕ) (stickers_per_other : ℕ) (leftover_stickers : ℕ) (total_students : ℕ) :
  total_stickers = 50 →
  friends = 5 →
  other_students = total_students - friends - 1 →
  stickers_per_other = 2 →
  leftover_stickers = 8 →
  total_students = 17 →
  (total_stickers - leftover_stickers - other_students * stickers_per_other) / friends = 4 := by
  sorry

#check marys_stickers

end NUMINAMATH_CALUDE_marys_stickers_l2355_235502


namespace NUMINAMATH_CALUDE_literary_readers_count_l2355_235523

theorem literary_readers_count (total : ℕ) (sci_fi : ℕ) (both : ℕ) (lit : ℕ) :
  total = 400 →
  sci_fi = 250 →
  both = 80 →
  total = sci_fi + lit - both →
  lit = 230 := by
sorry

end NUMINAMATH_CALUDE_literary_readers_count_l2355_235523


namespace NUMINAMATH_CALUDE_inequality_range_of_a_l2355_235506

theorem inequality_range_of_a (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) → 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_of_a_l2355_235506


namespace NUMINAMATH_CALUDE_sqrt_81_div_3_l2355_235539

theorem sqrt_81_div_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_div_3_l2355_235539


namespace NUMINAMATH_CALUDE_total_weight_of_cans_l2355_235545

theorem total_weight_of_cans (weights : List ℕ) (h : weights = [444, 459, 454, 459, 454, 454, 449, 454, 459, 464]) : 
  weights.sum = 4550 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_cans_l2355_235545


namespace NUMINAMATH_CALUDE_triangle_formation_l2355_235525

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 6 ∧ c = 9 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l2355_235525


namespace NUMINAMATH_CALUDE_water_volume_in_first_solution_l2355_235573

/-- The cost per liter of a spirit-water solution is directly proportional to the fraction of spirit by volume. -/
axiom cost_proportional_to_spirit_fraction (cost spirit_vol total_vol : ℝ) : 
  cost = (spirit_vol / total_vol) * (cost * total_vol / spirit_vol)

/-- The cost of the first solution with 1 liter of spirit and an unknown amount of water -/
def first_solution_cost : ℝ := 0.50

/-- The cost of the second solution with 1 liter of spirit and 2 liters of water -/
def second_solution_cost : ℝ := 0.50

/-- The volume of spirit in both solutions -/
def spirit_volume : ℝ := 1

/-- The volume of water in the second solution -/
def second_solution_water_volume : ℝ := 2

/-- The volume of water in the first solution -/
def first_solution_water_volume : ℝ := 2

theorem water_volume_in_first_solution : 
  first_solution_water_volume = 2 := by sorry

end NUMINAMATH_CALUDE_water_volume_in_first_solution_l2355_235573


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l2355_235538

/-- Given the room charges for three hotels P, R, and G, prove that R's charge is 170% greater than G's. -/
theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.7 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l2355_235538


namespace NUMINAMATH_CALUDE_bert_sale_earnings_l2355_235558

/-- Calculates Bert's earnings from a sale given the selling price, markup, and tax rate. -/
def bertEarnings (sellingPrice markup taxRate : ℚ) : ℚ :=
  let purchasePrice := sellingPrice - markup
  let tax := taxRate * sellingPrice
  sellingPrice - tax - purchasePrice

/-- Theorem: Given a selling price of $90, a markup of $10, and a tax rate of 10%, Bert's earnings are $1. -/
theorem bert_sale_earnings :
  bertEarnings 90 10 (1/10) = 1 := by
  sorry

#eval bertEarnings 90 10 (1/10)

end NUMINAMATH_CALUDE_bert_sale_earnings_l2355_235558


namespace NUMINAMATH_CALUDE_h_3_value_l2355_235576

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x + 9
def g (x : ℝ) : ℝ := (f x) ^ (1/3) - 3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_3_value : h 3 = 2 * 15^(1/3) + 3 := by sorry

end NUMINAMATH_CALUDE_h_3_value_l2355_235576


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l2355_235575

theorem relationship_between_exponents (a b c d x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(4*y) = a^(3*z)) 
  (h4 : c^(4*y) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  9*q*z = 8*x*y := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l2355_235575


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2355_235507

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := 5, y := 0, z := -1 }
  let a : Plane3D := { a := 2, b := -1, c := 3, d := -1 }
  let k : ℝ := 3
  let a' : Plane3D := transformPlane a k
  ¬ pointOnPlane A a' := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2355_235507


namespace NUMINAMATH_CALUDE_range_of_b_l2355_235571

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.sqrt (9 - p.1^2)}
def N (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + b}

-- State the theorem
theorem range_of_b (b : ℝ) : M ∩ N b = ∅ ↔ b > 3 * Real.sqrt 2 ∨ b < -3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l2355_235571


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2355_235595

/-- The coefficient of x^2 in the expansion of (1+2x)^5 is 40 -/
theorem coefficient_x_squared_in_expansion : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * 2^k * if k = 2 then 1 else 0) = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2355_235595


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2355_235582

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Define the solution set types
inductive SolutionSet
  | Interval
  | AllReals
  | Empty

-- State the theorem
theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0 ↔ x < -3 ∨ x > -2) → k = -2/5 ∧
  (∀ x, f k x < 0) → k < -Real.sqrt 6 / 6 ∧
  (∀ x, f k x ≥ 0) → k ≥ Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2355_235582


namespace NUMINAMATH_CALUDE_locus_of_center_C_l2355_235596

/-- Circle C₁ with equation x² + y² + 4y + 3 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.2 + 3 = 0}

/-- Circle C₂ with equation x² + y² - 4y - 77 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 - 77 = 0}

/-- The locus of the center of circle C -/
def locus_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 25 + p.1^2 / 21 = 1}

/-- Theorem stating that the locus of the center of circle C forms an ellipse
    given the tangency conditions with C₁ and C₂ -/
theorem locus_of_center_C (C : Set (ℝ × ℝ)) :
  (∃ r : ℝ, ∀ p ∈ C, ∃ q ∈ C₁, ‖p - q‖ = r) →  -- C is externally tangent to C₁
  (∃ R : ℝ, ∀ p ∈ C, ∃ q ∈ C₂, ‖p - q‖ = R) →  -- C is internally tangent to C₂
  C = locus_C :=
sorry

end NUMINAMATH_CALUDE_locus_of_center_C_l2355_235596


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2355_235501

/-- A geometric sequence {a_n} satisfying the given conditions has the specified general term. -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence condition
  a 1 + a 3 = 10 →                        -- First given condition
  a 2 + a 4 = 5 →                         -- Second given condition
  ∀ n, a n = 8 * (1/2)^(n - 1) :=         -- Conclusion: general term
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2355_235501


namespace NUMINAMATH_CALUDE_star_transformation_l2355_235580

theorem star_transformation (a b c d : ℕ) :
  a ∈ Finset.range 17 → b ∈ Finset.range 17 → c ∈ Finset.range 17 → d ∈ Finset.range 17 →
  a + b + c + d = 34 →
  (17 - a) + (17 - b) + (17 - c) + (17 - d) = 34 := by
sorry

end NUMINAMATH_CALUDE_star_transformation_l2355_235580


namespace NUMINAMATH_CALUDE_probability_calculations_l2355_235542

/-- Represents the number of students choosing each subject -/
structure SubjectCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  politics : ℕ
  history : ℕ
  geography : ℕ

/-- The total number of students -/
def totalStudents : ℕ := 1000

/-- The actual distribution of students across subjects -/
def actualCounts : SubjectCounts :=
  { physics := 300
  , chemistry := 200
  , biology := 100
  , politics := 200
  , history := 100
  , geography := 100 }

/-- Calculates the probability of an event given the number of favorable outcomes -/
def probability (favorableOutcomes : ℕ) : ℚ :=
  favorableOutcomes / totalStudents

/-- Theorem stating the probabilities of various events -/
theorem probability_calculations (counts : SubjectCounts) 
    (h : counts = actualCounts) : 
    probability counts.chemistry = 1/5 ∧ 
    probability (counts.biology + counts.history) = 1/5 ∧
    probability (counts.chemistry + counts.geography) = 3/10 := by
  sorry


end NUMINAMATH_CALUDE_probability_calculations_l2355_235542


namespace NUMINAMATH_CALUDE_droid_coffee_usage_l2355_235598

/-- The number of bags of coffee beans Droid uses in a week -/
def weekly_coffee_usage : ℕ :=
  let morning_usage := 3
  let afternoon_usage := 3 * morning_usage
  let evening_usage := 2 * morning_usage
  let daily_usage := morning_usage + afternoon_usage + evening_usage
  7 * daily_usage

/-- Theorem stating that Droid uses 126 bags of coffee beans in a week -/
theorem droid_coffee_usage : weekly_coffee_usage = 126 := by
  sorry

end NUMINAMATH_CALUDE_droid_coffee_usage_l2355_235598


namespace NUMINAMATH_CALUDE_train_length_l2355_235532

/-- The length of a train given its crossing times over a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ) 
  (h1 : bridge_length = 1500)
  (h2 : bridge_time = 70)
  (h3 : lamp_time = 20) :
  ∃ (train_length : ℝ), 
    train_length / lamp_time = (train_length + bridge_length) / bridge_time ∧ 
    train_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2355_235532


namespace NUMINAMATH_CALUDE_five_dice_probability_l2355_235592

/-- A die is represented as a number from 1 to 6 -/
def Die := Fin 6

/-- A roll of five dice -/
def FiveDiceRoll := Fin 5 → Die

/-- The probability space of rolling five fair six-sided dice -/
def Ω : Type := FiveDiceRoll

/-- The probability measure on Ω -/
noncomputable def P : Set Ω → ℝ := sorry

/-- The event that at least three dice show the same value -/
def AtLeastThreeSame (roll : Ω) : Prop := sorry

/-- The sum of the values shown on all dice -/
def DiceSum (roll : Ω) : ℕ := sorry

/-- The event that the sum of all dice is greater than 20 -/
def SumGreaterThan20 (roll : Ω) : Prop := DiceSum roll > 20

/-- The main theorem to be proved -/
theorem five_dice_probability : 
  P {roll : Ω | AtLeastThreeSame roll ∧ SumGreaterThan20 roll} = 31 / 432 := by sorry

end NUMINAMATH_CALUDE_five_dice_probability_l2355_235592


namespace NUMINAMATH_CALUDE_equal_profit_loss_price_correct_l2355_235519

/-- The selling price that results in equal profit and loss -/
def equalProfitLossPrice (costPrice : ℕ) (lossPrice : ℕ) : ℕ :=
  costPrice + (costPrice - lossPrice)

theorem equal_profit_loss_price_correct (costPrice lossPrice : ℕ) :
  let sellingPrice := equalProfitLossPrice costPrice lossPrice
  (sellingPrice - costPrice) = (costPrice - lossPrice) → sellingPrice = 57 :=
by
  intro sellingPrice h
  sorry

#eval equalProfitLossPrice 50 43

end NUMINAMATH_CALUDE_equal_profit_loss_price_correct_l2355_235519


namespace NUMINAMATH_CALUDE_power_relation_l2355_235535

theorem power_relation (a : ℝ) (m n : ℤ) (h1 : a^m = 3) (h2 : a^n = 2) :
  a^(m - 2*n) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_power_relation_l2355_235535


namespace NUMINAMATH_CALUDE_only_131_not_in_second_column_l2355_235584

def second_column (n : ℕ) : ℕ := 3 * n + 1

theorem only_131_not_in_second_column :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 400 →
    (31 = second_column n ∨
     94 = second_column n ∨
     331 = second_column n ∨
     907 = second_column n) ∧
    ¬(131 = second_column n) := by
  sorry

end NUMINAMATH_CALUDE_only_131_not_in_second_column_l2355_235584


namespace NUMINAMATH_CALUDE_find_B_l2355_235585

theorem find_B : ∃ B : ℕ, 
  (632 - 591 = 41) ∧ 
  (∃ (AB1 : ℕ), AB1 = 500 + 90 + B ∧ AB1 < 1000) → 
  B = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l2355_235585


namespace NUMINAMATH_CALUDE_equation_solution_l2355_235534

theorem equation_solution :
  let f (x : ℚ) := (3 - x) / (x + 2) + (3*x - 6) / (3 - x)
  ∃! x, f x = 2 ∧ x = -7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2355_235534


namespace NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_l2355_235563

theorem integral_sqrt_4_minus_x_squared : ∫ x in (-2)..2, Real.sqrt (4 - x^2) = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_l2355_235563


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_20_l2355_235567

theorem least_product_of_distinct_primes_above_20 :
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 20 ∧ q > 20 ∧ 
    p ≠ q ∧
    p * q = 667 ∧
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 20 → s > 20 → r ≠ s → r * s ≥ 667 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_20_l2355_235567


namespace NUMINAMATH_CALUDE_range_of_m_l2355_235594

theorem range_of_m : 
  (∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) → 
  (∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2355_235594


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2355_235557

def f (x : ℝ) : ℝ := 3 * x^2 + 8 * x - 10

theorem f_increasing_on_interval : 
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2355_235557


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l2355_235513

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 250 / 98 →
  ∃ (a b c : ℕ), 
    (a = 25 ∧ b = 5 ∧ c = 7) ∧
    (Real.sqrt area_ratio * c = a * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l2355_235513


namespace NUMINAMATH_CALUDE_max_value_abc_expression_l2355_235527

theorem max_value_abc_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_expression_l2355_235527


namespace NUMINAMATH_CALUDE_science_marks_calculation_l2355_235565

def average_marks : ℝ := 75
def num_subjects : ℕ := 5
def math_marks : ℝ := 76
def social_marks : ℝ := 82
def english_marks : ℝ := 67
def biology_marks : ℝ := 85

theorem science_marks_calculation :
  ∃ (science_marks : ℝ),
    (math_marks + social_marks + english_marks + biology_marks + science_marks) / num_subjects = average_marks ∧
    science_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_science_marks_calculation_l2355_235565


namespace NUMINAMATH_CALUDE_no_valid_replacements_l2355_235572

theorem no_valid_replacements :
  ∀ z : ℕ, z < 10 → ¬(35000 + 100 * z + 45) % 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_valid_replacements_l2355_235572


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l2355_235536

theorem divisible_by_thirteen (a b : ℕ) (h : a * 13 = 119268916) :
  119268903 % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l2355_235536


namespace NUMINAMATH_CALUDE_lottery_savings_calculation_l2355_235556

theorem lottery_savings_calculation (lottery_winnings : ℚ) 
  (tax_rate : ℚ) (student_loan_rate : ℚ) (investment_rate : ℚ) (fun_money : ℚ) :
  lottery_winnings = 12006 →
  tax_rate = 1/2 →
  student_loan_rate = 1/3 →
  investment_rate = 1/5 →
  fun_money = 2802 →
  ∃ (savings : ℚ),
    savings = 1000 ∧
    lottery_winnings * (1 - tax_rate) * (1 - student_loan_rate) - fun_money = savings * (1 + investment_rate) :=
by sorry

end NUMINAMATH_CALUDE_lottery_savings_calculation_l2355_235556


namespace NUMINAMATH_CALUDE_elevator_occupancy_l2355_235561

/-- Proves that the total number of people in the elevator is 7 after a new person enters --/
theorem elevator_occupancy (initial_people : ℕ) (initial_avg_weight : ℝ) (new_avg_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 160 →
  new_avg_weight = 151 →
  initial_people + 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_elevator_occupancy_l2355_235561
