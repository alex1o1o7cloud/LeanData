import Mathlib

namespace shortest_side_length_l1413_141372

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The length of the first segment of the partitioned side -/
  segment1 : ℝ
  /-- The length of the second segment of the partitioned side -/
  segment2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The angle opposite the partitioned side in radians -/
  opposite_angle : ℝ

/-- The theorem stating the length of the shortest side of the triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle)
  (h1 : t.segment1 = 7)
  (h2 : t.segment2 = 9)
  (h3 : t.radius = 5)
  (h4 : t.opposite_angle = π / 3) :
  ∃ (shortest_side : ℝ), shortest_side = 20 * (2 + Real.sqrt 3) := by
  sorry

end shortest_side_length_l1413_141372


namespace min_value_expression_l1413_141320

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (min : ℝ), min = Real.sqrt 6 ∧
  ∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0),
    x^2 + 2*y^2 + 1/x^2 + 2*y/x ≥ min ∧
    ∃ (a₀ b₀ : ℝ) (ha₀ : a₀ ≠ 0) (hb₀ : b₀ ≠ 0),
      a₀^2 + 2*b₀^2 + 1/a₀^2 + 2*b₀/a₀ = min :=
by sorry

end min_value_expression_l1413_141320


namespace ellipse_chord_fixed_point_l1413_141364

/-- The fixed point theorem for ellipse chords -/
theorem ellipse_chord_fixed_point 
  (a b A B : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hAB : A ≠ 0 ∧ B ≠ 0) :
  ∃ M : ℝ × ℝ, ∀ P : ℝ × ℝ,
    (A * P.1 + B * P.2 = 1) →  -- P is on line l
    ∃ Q R : ℝ × ℝ,
      (R.1^2 / a^2 + R.2^2 / b^2 = 1) ∧  -- R is on ellipse Γ
      (∃ t : ℝ, Q = ⟨t * P.1, t * P.2⟩) ∧  -- Q is on ray OP
      (Q.1^2 + Q.2^2) * (P.1^2 + P.2^2) = (R.1^2 + R.2^2)^2 →  -- |OQ| * |OP| = |OR|^2
      ∃ l_P : Set (ℝ × ℝ),
        (∀ X ∈ l_P, ∃ s : ℝ, X = ⟨s * Q.1, s * Q.2⟩) ∧  -- l_P is a line through Q
        M ∈ l_P ∧  -- M is on l_P
        M = (A * a^2, B * b^2) :=
by
  sorry

end ellipse_chord_fixed_point_l1413_141364


namespace concyclic_intersection_points_l1413_141378

structure Circle where
  center : Point
  radius : ℝ

structure Chord (c : Circle) where
  endpoint1 : Point
  endpoint2 : Point

def midpoint_of_arc (c : Circle) (ch : Chord c) : Point := sorry

def intersect_chords (c : Circle) (ch1 ch2 : Chord c) : Point := sorry

def concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

theorem concyclic_intersection_points 
  (c : Circle) 
  (bc : Chord c) 
  (a : Point) 
  (ad ae : Chord c) 
  (f g : Point) :
  a = midpoint_of_arc c bc →
  f = intersect_chords c bc ad →
  g = intersect_chords c bc ae →
  concyclic (ad.endpoint2) (ae.endpoint2) f g :=
sorry

end concyclic_intersection_points_l1413_141378


namespace smallest_x_for_equation_l1413_141303

theorem smallest_x_for_equation : 
  ∀ x : ℝ, x > 0 → (⌊x^2⌋ : ℤ) - x * (⌊x⌋ : ℤ) = 10 → x ≥ 131/11 :=
by sorry

end smallest_x_for_equation_l1413_141303


namespace markers_given_l1413_141360

theorem markers_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 217 → total = 326 → given = total - initial → given = 109 := by
sorry

end markers_given_l1413_141360


namespace fraction_sum_l1413_141340

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 2) : (a + b) / b = 5 / 2 := by
  sorry

end fraction_sum_l1413_141340


namespace circle_C_equation_l1413_141361

-- Define the circles and line
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2
def line_symmetry (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the symmetry condition
def symmetric_circles (C_center : ℝ × ℝ) (r : ℝ) : Prop :=
  let (a, b) := C_center
  (a - (-2)) / 2 + (b - (-2)) / 2 + 2 = 0 ∧ (b + 2) / (a + 2) = 1

-- Theorem statement
theorem circle_C_equation :
  ∀ (r : ℝ), r > 0 →
  ∃ (C_center : ℝ × ℝ),
    (symmetric_circles C_center r) ∧
    ((1 : ℝ) - C_center.1)^2 + ((1 : ℝ) - C_center.2)^2 = 
    C_center.1^2 + C_center.2^2 →
    ∀ (x y : ℝ), x^2 + y^2 = 2 ↔ 
      ((x - C_center.1)^2 + (y - C_center.2)^2 = C_center.1^2 + C_center.2^2) :=
by
  sorry


end circle_C_equation_l1413_141361


namespace product_polynomial_sum_l1413_141381

theorem product_polynomial_sum (g h : ℚ) : 
  (∀ d : ℚ, (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 5) = 
   21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) →
  g + h = -28/9 := by
sorry

end product_polynomial_sum_l1413_141381


namespace percentage_problem_l1413_141337

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1600 - 15 → x = 900 := by
  sorry

end percentage_problem_l1413_141337


namespace genevieve_coffee_consumption_l1413_141318

-- Define the conversion rate from gallons to pints
def gallons_to_pints (gallons : Real) : Real := gallons * 8

-- Define the total amount of coffee in gallons
def total_coffee_gallons : Real := 4.5

-- Define the number of thermoses
def num_thermoses : Nat := 18

-- Define the number of thermoses Genevieve drank
def genevieve_thermoses : Nat := 3

-- Theorem statement
theorem genevieve_coffee_consumption :
  let total_pints := gallons_to_pints total_coffee_gallons
  let pints_per_thermos := total_pints / num_thermoses
  pints_per_thermos * genevieve_thermoses = 6 := by
  sorry


end genevieve_coffee_consumption_l1413_141318


namespace min_value_sum_product_l1413_141355

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end min_value_sum_product_l1413_141355


namespace count_integers_with_8_and_9_between_700_and_1000_l1413_141396

def count_integers_with_8_and_9 (lower_bound upper_bound : ℕ) : ℕ :=
  (upper_bound - lower_bound + 1) / 100 * 2

theorem count_integers_with_8_and_9_between_700_and_1000 :
  count_integers_with_8_and_9 700 1000 = 6 := by
  sorry

end count_integers_with_8_and_9_between_700_and_1000_l1413_141396


namespace work_time_for_less_efficient_worker_l1413_141330

/-- Represents the time it takes for a worker to complete a job alone -/
def WorkTime := ℝ

/-- Represents the efficiency of a worker (fraction of job completed per day) -/
def Efficiency := ℝ

theorem work_time_for_less_efficient_worker 
  (total_time : ℝ) 
  (efficiency_ratio : ℝ) :
  total_time > 0 →
  efficiency_ratio > 1 →
  let joint_efficiency := 1 / total_time
  let less_efficient_worker_efficiency := joint_efficiency / (1 + efficiency_ratio)
  let work_time_less_efficient := 1 / less_efficient_worker_efficiency
  (total_time = 36 ∧ efficiency_ratio = 2) → work_time_less_efficient = 108 := by
  sorry

end work_time_for_less_efficient_worker_l1413_141330


namespace initial_speed_is_five_l1413_141311

/-- Proves that the initial speed is 5 km/hr given the conditions of the journey --/
theorem initial_speed_is_five (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 26.67)
  (h2 : total_time = 6)
  (h3 : second_half_speed = 4)
  (h4 : (total_distance / 2) / v + (total_distance / 2) / second_half_speed = total_time)
  : v = 5 := by
  sorry

end initial_speed_is_five_l1413_141311


namespace deck_cost_l1413_141353

/-- The cost of the deck of playing cards given the allowances and sticker purchases -/
theorem deck_cost (lola_allowance dora_allowance : ℕ)
                  (sticker_boxes : ℕ)
                  (dora_sticker_packs : ℕ)
                  (h1 : lola_allowance = 9)
                  (h2 : dora_allowance = 9)
                  (h3 : sticker_boxes = 2)
                  (h4 : dora_sticker_packs = 2) :
  let total_allowance := lola_allowance + dora_allowance
  let total_sticker_packs := 2 * dora_sticker_packs
  let sticker_cost := sticker_boxes * 2
  total_allowance - sticker_cost = 10 := by
sorry

end deck_cost_l1413_141353


namespace floor_length_proof_l1413_141332

/-- Proves that the length of a rectangular floor is 24 meters given specific conditions -/
theorem floor_length_proof (width : ℝ) (square_size : ℝ) (total_cost : ℝ) (square_cost : ℝ) :
  width = 64 →
  square_size = 8 →
  total_cost = 576 →
  square_cost = 24 →
  (total_cost / square_cost) * square_size * square_size / width = 24 := by
sorry

end floor_length_proof_l1413_141332


namespace expansion_properties_l1413_141384

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion term
def expansionTerm (n r : ℕ) (x : ℝ) : ℝ := 
  (binomial n r : ℝ) * (2^(n-r)) * (3^r) * (x^(n - (4/3)*r))

theorem expansion_properties :
  ∃ (n : ℕ) (x : ℝ),
  -- Condition: ratio of binomial coefficients
  (binomial n 2 : ℝ) / (binomial n 1 : ℝ) = 5/2 →
  -- 1. n = 6
  n = 6 ∧
  -- 2. Coefficient of x^2 term
  (∃ (r : ℕ), n - (4/3)*r = 2 ∧ 
    expansionTerm n r 1 = 4320) ∧
  -- 3. Term with maximum coefficient
  (∃ (r : ℕ), ∀ (k : ℕ), 
    expansionTerm n r x ≥ expansionTerm n k x ∧
    expansionTerm n r 1 = 4860 ∧
    n - (4/3)*r = 2/3) :=
sorry

end expansion_properties_l1413_141384


namespace money_distribution_l1413_141358

theorem money_distribution (total money_ac money_bc : ℕ) 
  (h1 : total = 600)
  (h2 : money_ac = 250)
  (h3 : money_bc = 450) :
  ∃ (a b c : ℕ), a + b + c = total ∧ a + c = money_ac ∧ b + c = money_bc ∧ c = 100 := by
  sorry

end money_distribution_l1413_141358


namespace basketball_count_l1413_141308

theorem basketball_count :
  ∀ (basketballs volleyballs soccerballs : ℕ),
    basketballs + volleyballs + soccerballs = 100 →
    basketballs = 2 * volleyballs →
    volleyballs = soccerballs + 8 →
    basketballs = 54 :=
by
  sorry

end basketball_count_l1413_141308


namespace partial_fraction_decomposition_l1413_141331

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ x : ℝ,
    (x^3 - 2*x^2 + x - 1) / (x^3 + 2*x^2 + x + 1) = 
    P / (x + 1) + (Q*x + R) / (x^2 + 1) ∧
    P = -2 ∧ Q = 0 ∧ R = 1 :=
by sorry

end partial_fraction_decomposition_l1413_141331


namespace xenia_june_earnings_l1413_141338

/-- Xenia's earnings during the first two weeks of June -/
def xenia_earnings (hours_week1 hours_week2 : ℕ) (wage_difference : ℚ) : ℚ :=
  let hourly_wage := wage_difference / (hours_week2 - hours_week1 : ℚ)
  hourly_wage * (hours_week1 + hours_week2 : ℚ)

/-- Theorem stating Xenia's earnings during the first two weeks of June -/
theorem xenia_june_earnings :
  xenia_earnings 15 22 (47.60 : ℚ) = (251.60 : ℚ) := by
  sorry

#eval xenia_earnings 15 22 (47.60 : ℚ)

end xenia_june_earnings_l1413_141338


namespace golden_section_length_l1413_141319

/-- Definition of a golden section point -/
def is_golden_section (A B C : ℝ) : Prop :=
  (B - A) / (C - A) = (C - A) / (A - C)

/-- Theorem: Length of AC when C is a golden section point of AB -/
theorem golden_section_length (A B C : ℝ) :
  B - A = 20 →
  is_golden_section A B C →
  (C - A = 10 * Real.sqrt 5 - 10) ∨ (C - A = 30 - 10 * Real.sqrt 5) :=
by sorry

end golden_section_length_l1413_141319


namespace angle_relation_in_triangle_l1413_141328

/-- Given a triangle XYZ with an interior point E, where a, b, c, p are the measures of angles
    around E in degrees, and t is the exterior angle at vertex Y, prove that p = 180° - a - b + t. -/
theorem angle_relation_in_triangle (a b c p t : ℝ) : 
  (a + b + c + p = 360) →  -- Sum of angles around interior point E
  (t = 180 - c) →          -- Exterior angle relation
  (p = 180 - a - b + t) :=
by sorry

end angle_relation_in_triangle_l1413_141328


namespace equation_solutions_l1413_141379

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

end equation_solutions_l1413_141379


namespace sum_of_coefficients_l1413_141327

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : ∃ a b c : ℝ, 
  (16 * d + 17 + 18 * d^3) + (4 * d + 2) = a * d^3 + b * d + c ∧ a + b + c = 57 := by
  sorry

end sum_of_coefficients_l1413_141327


namespace cube_difference_of_squares_l1413_141309

theorem cube_difference_of_squares (a : ℕ+) :
  ∃ (x y : ℤ), x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end cube_difference_of_squares_l1413_141309


namespace probability_of_star_is_one_fifth_l1413_141341

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)

/-- Calculates the probability of drawing a specific suit from a deck -/
def probability_of_suit (d : Deck) : ℚ :=
  d.cards_per_suit / d.total_cards

/-- The modified deck of cards as described in the problem -/
def modified_deck : Deck :=
  { total_cards := 65,
    num_suits := 5,
    cards_per_suit := 13 }

theorem probability_of_star_is_one_fifth :
  probability_of_suit modified_deck = 1 / 5 := by
  sorry

end probability_of_star_is_one_fifth_l1413_141341


namespace point_in_fourth_quadrant_l1413_141387

/-- If |a-4|+(b+3)^2=0, then a > 0 and b < 0 -/
theorem point_in_fourth_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : a > 0 ∧ b < 0 := by
  sorry

end point_in_fourth_quadrant_l1413_141387


namespace power_of_64_five_sixths_l1413_141371

theorem power_of_64_five_sixths : (64 : ℝ) ^ (5/6) = 32 := by sorry

end power_of_64_five_sixths_l1413_141371


namespace ellipse_C_properties_l1413_141385

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 (a > b > 0), 
    eccentricity √3/3, and major axis length 2√3 --/
def ellipse_C (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (a^2 - b^2) / a^2 = 3 / 9 ∧
  2 * a = 2 * Real.sqrt 3

/-- The equation of ellipse C --/
def ellipse_C_equation (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

/-- Circle O with major axis of ellipse C as its diameter --/
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 3

/-- Point on circle O --/
def point_on_circle_O (M : ℝ × ℝ) : Prop :=
  circle_O M.1 M.2

/-- Line perpendicular to OM passing through M --/
def perpendicular_line (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  M.1 * (x - M.1) + M.2 * (y - M.2) = 0

theorem ellipse_C_properties (a b : ℝ) (h : ellipse_C a b) :
  (∀ x y, ellipse_C_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∀ M : ℝ × ℝ, point_on_circle_O M →
    ∃ x y, perpendicular_line M x y ∧ x = 1 ∧ y = 0) :=
sorry

end ellipse_C_properties_l1413_141385


namespace harry_tomato_packets_l1413_141347

/-- Represents the number of packets of tomato seeds Harry bought -/
def tomato_packets : ℕ := sorry

/-- The price of a packet of pumpkin seeds in dollars -/
def pumpkin_price : ℚ := 5/2

/-- The price of a packet of tomato seeds in dollars -/
def tomato_price : ℚ := 3/2

/-- The price of a packet of chili pepper seeds in dollars -/
def chili_price : ℚ := 9/10

/-- The number of packets of pumpkin seeds Harry bought -/
def pumpkin_bought : ℕ := 3

/-- The number of packets of chili pepper seeds Harry bought -/
def chili_bought : ℕ := 5

/-- The total amount Harry spent in dollars -/
def total_spent : ℚ := 18

theorem harry_tomato_packets : 
  pumpkin_price * pumpkin_bought + tomato_price * tomato_packets + chili_price * chili_bought = total_spent ∧ 
  tomato_packets = 4 := by sorry

end harry_tomato_packets_l1413_141347


namespace total_students_is_28_l1413_141325

/-- The number of students taking the AMC 8 in Mrs. Germain's class -/
def germain_students : ℕ := 11

/-- The number of students taking the AMC 8 in Mr. Newton's class -/
def newton_students : ℕ := 8

/-- The number of students taking the AMC 8 in Mrs. Young's class -/
def young_students : ℕ := 9

/-- The total number of students taking the AMC 8 at Euclid Middle School -/
def total_students : ℕ := germain_students + newton_students + young_students

/-- Theorem stating that the total number of students taking the AMC 8 is 28 -/
theorem total_students_is_28 : total_students = 28 := by sorry

end total_students_is_28_l1413_141325


namespace min_cards_sum_eleven_l1413_141342

theorem min_cards_sum_eleven (n : ℕ) (h : n = 10) : 
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card ≥ k → 
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 11) ∧
  (∀ (m : ℕ), m < k → 
    ∃ (T : Finset ℕ), T ⊆ Finset.range n ∧ T.card = m ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → a + b ≠ 11) :=
by sorry

end min_cards_sum_eleven_l1413_141342


namespace positive_numbers_not_all_equal_l1413_141370

/-- Given positive numbers a, b, and c that are not all equal -/
theorem positive_numbers_not_all_equal 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  /- 1. (a-b)² + (b-c)² + (c-a)² ≠ 0 -/
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  /- 2. At least one of a > b, a < b, or a = b is true -/
  (a > b ∨ a < b ∨ a = b) ∧ 
  /- 3. It is possible for a ≠ c, b ≠ c, and a ≠ b to all be true simultaneously -/
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z := by
  sorry

end positive_numbers_not_all_equal_l1413_141370


namespace smallest_bob_number_l1413_141356

def alice_number : ℕ := 30

theorem smallest_bob_number (bob_number : ℕ) 
  (h1 : ∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ bob_number) 
  (h2 : ∀ n : ℕ, n ≥ bob_number → 
    (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n)) : 
  bob_number = 30 := by
sorry

end smallest_bob_number_l1413_141356


namespace chord_cosine_l1413_141301

theorem chord_cosine (r : ℝ) (θ φ : ℝ) : 
  r > 0 →
  θ > 0 →
  φ > 0 →
  θ + φ < π →
  8^2 = 2 * r^2 * (1 - Real.cos θ) →
  15^2 = 2 * r^2 * (1 - Real.cos φ) →
  17^2 = 2 * r^2 * (1 - Real.cos (θ + φ)) →
  Real.cos θ = 161 / 225 := by
sorry

end chord_cosine_l1413_141301


namespace imaginary_part_product_l1413_141343

theorem imaginary_part_product : Complex.im ((2 - Complex.I) * (1 - 2 * Complex.I)) = -5 := by sorry

end imaginary_part_product_l1413_141343


namespace point_on_x_axis_l1413_141335

theorem point_on_x_axis (m : ℝ) : (∃ P : ℝ × ℝ, P.1 = m + 5 ∧ P.2 = m - 2 ∧ P.2 = 0) → m = 2 := by
  sorry

end point_on_x_axis_l1413_141335


namespace steves_nickels_l1413_141397

theorem steves_nickels (nickels dimes : ℕ) : 
  dimes = nickels + 4 →
  5 * nickels + 10 * dimes = 70 →
  nickels = 2 := by
sorry

end steves_nickels_l1413_141397


namespace smallest_n_not_divisible_by_10_l1413_141334

theorem smallest_n_not_divisible_by_10 :
  ∃ (n : ℕ), n = 2020 ∧ n > 2016 ∧
  ¬(10 ∣ (1^n + 2^n + 3^n + 4^n)) ∧
  ∀ (m : ℕ), 2016 < m ∧ m < n → (10 ∣ (1^m + 2^m + 3^m + 4^m)) :=
by sorry

end smallest_n_not_divisible_by_10_l1413_141334


namespace string_displacement_impossible_l1413_141349

/-- A rectangular parallelepiped box with strings. -/
structure StringBox where
  a : ℝ
  b : ℝ
  c : ℝ
  N : ℝ × ℝ × ℝ
  P : ℝ × ℝ × ℝ

/-- Strings cross at right angles at N and P. -/
def strings_cross_at_right_angles (box : StringBox) : Prop :=
  sorry

/-- Strings are strongly glued at N and P. -/
def strings_strongly_glued (box : StringBox) : Prop :=
  sorry

/-- Any displacement of the strings is impossible. -/
def no_displacement_possible (box : StringBox) : Prop :=
  sorry

/-- Theorem: If strings cross at right angles and are strongly glued at N and P,
    then any displacement of the strings is impossible. -/
theorem string_displacement_impossible (box : StringBox) :
  strings_cross_at_right_angles box →
  strings_strongly_glued box →
  no_displacement_possible box :=
by
  sorry

end string_displacement_impossible_l1413_141349


namespace trees_in_row_l1413_141329

/-- Given a plot of trees with the following properties:
  1. Trees are planted in rows of 4
  2. Each tree gives 5 apples
  3. Each apple is sold for $0.5
  4. Total revenue is $30
  Prove that the number of trees in one row is 4. -/
theorem trees_in_row (trees_per_row : ℕ) (apples_per_tree : ℕ) (price_per_apple : ℚ) (total_revenue : ℚ)
  (h1 : trees_per_row = 4)
  (h2 : apples_per_tree = 5)
  (h3 : price_per_apple = 1/2)
  (h4 : total_revenue = 30) :
  trees_per_row = 4 := by sorry

end trees_in_row_l1413_141329


namespace train_journey_time_l1413_141345

theorem train_journey_time 
  (D : ℝ) -- Distance in km
  (T : ℝ) -- Original time in hours
  (h1 : D = 48 * T) -- Distance equation for original journey
  (h2 : D = 60 * (40 / 60)) -- Distance equation for faster journey
  : T * 60 = 50 := by
  sorry

end train_journey_time_l1413_141345


namespace phone_number_probability_l1413_141326

/-- Represents the possible prefixes for the phone number -/
def prefixes : Finset String := {"296", "299", "298"}

/-- Represents the digits for the remaining part of the phone number -/
def remainingDigits : Finset Char := {'0', '1', '6', '7', '9'}

/-- The total number of digits in the phone number -/
def totalDigits : Nat := 8

theorem phone_number_probability :
  (Finset.card prefixes * (Finset.card remainingDigits).factorial : ℚ)⁻¹ = 1 / 360 := by
  sorry

end phone_number_probability_l1413_141326


namespace geometry_theorem_l1413_141366

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (distinct : Line → Line → Prop)
variable (distinctP : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem geometry_theorem 
  (h_distinct_lines : distinct m n)
  (h_distinct_planes : distinctP α β) :
  (perpendicularPP α β ∧ perpendicularLP m α → ¬(intersects m β)) ∧
  (perpendicular m n ∧ perpendicularLP m α → ¬(intersects n α)) :=
sorry

end geometry_theorem_l1413_141366


namespace hyperbola_eccentricity_l1413_141323

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    where a > 0, b > 0, and one asymptote forms a 60° angle with the y-axis. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_asymptote : b / a = Real.sqrt 3 / 3) : 
    Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l1413_141323


namespace ramon_twice_loui_age_l1413_141336

/-- The age of Loui today -/
def loui_age : ℕ := 23

/-- The age of Ramon today -/
def ramon_age : ℕ := 26

/-- The number of years until Ramon is twice as old as Loui is today -/
def years_until_double : ℕ := 20

/-- Theorem stating that in 'years_until_double' years, Ramon will be twice as old as Loui is today -/
theorem ramon_twice_loui_age : 
  ramon_age + years_until_double = 2 * loui_age := by
  sorry

end ramon_twice_loui_age_l1413_141336


namespace justin_flower_gathering_time_l1413_141350

/-- Proves that Justin has been gathering flowers for 1 hour given the problem conditions -/
theorem justin_flower_gathering_time :
  let classmates : ℕ := 30
  let time_per_flower : ℕ := 10  -- minutes
  let lost_flowers : ℕ := 3
  let remaining_time : ℕ := 210  -- minutes
  let total_flowers_needed : ℕ := classmates
  let remaining_flowers : ℕ := remaining_time / time_per_flower + lost_flowers
  let gathered_flowers : ℕ := total_flowers_needed - remaining_flowers
  let gathering_time : ℕ := gathered_flowers * time_per_flower
  gathering_time / 60 = 1  -- hours
  := by sorry

end justin_flower_gathering_time_l1413_141350


namespace divisible_by_27_l1413_141310

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) - 9*n^2 + 21*n - 14 = 27*k := by
  sorry

end divisible_by_27_l1413_141310


namespace parabola_symmetry_l1413_141399

/-- Given two parabolas, prove that they are symmetrical about the x-axis -/
theorem parabola_symmetry (x : ℝ) : 
  let f (x : ℝ) := (x - 1)^2 + 3
  let g (x : ℝ) := -(x - 1)^2 - 3
  ∀ x, f x = -g x := by
  sorry

end parabola_symmetry_l1413_141399


namespace negation_of_proposition_l1413_141380

theorem negation_of_proposition :
  (¬ (∀ n : ℕ, n^2 < 3*n + 4)) ↔ (∃ n : ℕ, n^2 ≥ 3*n + 4) := by
  sorry

end negation_of_proposition_l1413_141380


namespace geometric_mean_problem_l1413_141374

theorem geometric_mean_problem (a b c : ℝ) 
  (h1 : b^2 = a*c)  -- b is the geometric mean of a and c
  (h2 : a*b*c = 27) : b = 3 := by
  sorry

end geometric_mean_problem_l1413_141374


namespace kangaroo_jump_distance_l1413_141392

/-- Proves that a kangaroo jumping up and down a mountain with specific jump patterns covers a total distance of 3036 meters. -/
theorem kangaroo_jump_distance (total_jumps : ℕ) (uphill_distance downhill_distance : ℝ) 
  (h1 : total_jumps = 2024)
  (h2 : uphill_distance = 1)
  (h3 : downhill_distance = 3)
  (h4 : ∃ (uphill_jumps downhill_jumps : ℕ), 
    uphill_jumps + downhill_jumps = total_jumps ∧ 
    uphill_jumps = 3 * downhill_jumps) :
  ∃ (total_distance : ℝ), total_distance = 3036 := by
  sorry

end kangaroo_jump_distance_l1413_141392


namespace lcm_primes_sum_l1413_141305

theorem lcm_primes_sum (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x > y → Nat.lcm x y = 10 → 2 * x + y = 12 := by
  sorry

end lcm_primes_sum_l1413_141305


namespace num_possible_strings_l1413_141304

/-- Represents the allowed moves in the string transformation game -/
inductive Move
| HM_to_MH
| MT_to_TM
| TH_to_HT

/-- The initial string in the game -/
def initial_string : String := "HHMMMMTT"

/-- The number of H's in the initial string -/
def num_H : Nat := 2

/-- The number of M's in the initial string -/
def num_M : Nat := 4

/-- The number of T's in the initial string -/
def num_T : Nat := 2

/-- The total length of the string -/
def total_length : Nat := num_H + num_M + num_T

/-- Theorem stating that the number of possible strings after zero or more moves
    is equal to the number of ways to choose num_M positions out of total_length positions -/
theorem num_possible_strings :
  (Nat.choose total_length num_M) = 70 := by sorry

end num_possible_strings_l1413_141304


namespace max_largest_integer_l1413_141314

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 70 →
  e - a = 10 →
  a < b ∧ b < c ∧ c < d ∧ d < e →
  e ≤ 340 :=
sorry

end max_largest_integer_l1413_141314


namespace triangle_reconstruction_l1413_141393

-- Define the centers of the squares
structure SquareCenters where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  O₃ : ℝ × ℝ

-- Define a 90-degree rotation around a point
def rotate90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the composition of rotations
def compositeRotation (centers : SquareCenters) (point : ℝ × ℝ) : ℝ × ℝ :=
  rotate90 centers.O₃ (rotate90 centers.O₂ (rotate90 centers.O₁ point))

-- Theorem stating the existence of an invariant point
theorem triangle_reconstruction (centers : SquareCenters) :
  ∃ (B : ℝ × ℝ), compositeRotation centers B = B :=
sorry

end triangle_reconstruction_l1413_141393


namespace f_properties_l1413_141346

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

theorem f_properties :
  (is_smallest_positive_period f π) ∧
  (∀ x, 1/2 ≤ f x ∧ f x ≤ 5/2) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-π/3 + k*π) (π/6 + k*π))) :=
sorry

end f_properties_l1413_141346


namespace gas_cost_theorem_l1413_141363

/-- Calculates the total cost of gas for Ryosuke's travels in a day -/
def calculate_gas_cost (trip1_start trip1_end trip2_start trip2_end : ℕ) 
                       (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let total_distance : ℕ := (trip1_end - trip1_start) + (trip2_end - trip2_start)
  let gallons_used : ℚ := total_distance / fuel_efficiency
  let total_cost : ℚ := gallons_used * gas_price
  total_cost

/-- The total cost of gas for Ryosuke's travels is approximately $10.11 -/
theorem gas_cost_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |calculate_gas_cost 63102 63135 63135 63166 25 (395/100) - (1011/100)| < ε :=
sorry

end gas_cost_theorem_l1413_141363


namespace half_plus_five_equals_fifteen_l1413_141333

theorem half_plus_five_equals_fifteen (n : ℕ) (value : ℕ) : n = 20 → n / 2 + 5 = value → value = 15 := by
  sorry

end half_plus_five_equals_fifteen_l1413_141333


namespace intersection_M_N_l1413_141344

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 = x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l1413_141344


namespace triple_equality_l1413_141375

theorem triple_equality (a b c : ℝ) : 
  a * (b^2 + c) = c * (c + a * b) →
  b * (c^2 + a) = a * (a + b * c) →
  c * (a^2 + b) = b * (b + c * a) →
  a = b ∧ b = c :=
by sorry

end triple_equality_l1413_141375


namespace geometric_sequence_fifth_term_l1413_141383

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a2 : a 2 = 2)
  (h_sum : 2 * a 3 + a 4 = 16) :
  a 5 = 16 := by
  sorry

end geometric_sequence_fifth_term_l1413_141383


namespace pure_imaginary_complex_number_l1413_141315

theorem pure_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)
  (z.re = 0 ∧ z.im ≠ 0) → m = -2 :=
by sorry

end pure_imaginary_complex_number_l1413_141315


namespace als_original_portion_l1413_141324

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  0.75 * a + 2 * b + 2 * c = 1800 →
  a = 480 :=
by sorry

end als_original_portion_l1413_141324


namespace real_z_implies_m_eq_3_modulus_z_eq_sqrt_13_when_m_eq_1_l1413_141321

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m + 2) (m - 3)

-- Theorem 1: If z is a real number, then m = 3
theorem real_z_implies_m_eq_3 (m : ℝ) : z m = Complex.mk (z m).re 0 → m = 3 := by
  sorry

-- Theorem 2: When m = 1, the modulus of z is √13
theorem modulus_z_eq_sqrt_13_when_m_eq_1 : Complex.abs (z 1) = Real.sqrt 13 := by
  sorry

end real_z_implies_m_eq_3_modulus_z_eq_sqrt_13_when_m_eq_1_l1413_141321


namespace cosine_sine_sum_zero_l1413_141317

theorem cosine_sine_sum_zero (x : ℝ) (h : Real.cos (π / 6 - x) = -Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 := by
  sorry

end cosine_sine_sum_zero_l1413_141317


namespace system_solution_l1413_141373

theorem system_solution (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.sin a - x₁ * Real.cos a) ∧
     x₁^2 + y₁^2 = 26 * (y₁ * Real.cos (2*a) - x₁ * Real.sin (2*a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.sin a - x₂ * Real.cos a) ∧
     x₂^2 + y₂^2 = 26 * (y₂ * Real.cos (2*a) - x₂ * Real.sin (2*a))) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 24^2) →
  (∃ n : ℤ, a = π/6 + (2/3) * Real.arctan (5/12) + (2*π*n)/3 ∨
            a = π/6 - (2/3) * Real.arctan (5/12) + (2*π*n)/3 ∨
            a = π/6 + (2*π*n)/3) :=
by sorry

end system_solution_l1413_141373


namespace committee_selection_ways_l1413_141394

-- Define the total number of members
def total_members : ℕ := 30

-- Define the number of ineligible members
def ineligible_members : ℕ := 3

-- Define the size of the committee
def committee_size : ℕ := 5

-- Define the number of eligible members
def eligible_members : ℕ := total_members - ineligible_members

-- Theorem statement
theorem committee_selection_ways :
  Nat.choose eligible_members committee_size = 80730 := by
  sorry

end committee_selection_ways_l1413_141394


namespace second_reduction_percentage_store_price_reduction_l1413_141382

/-- Given two successive price reductions, calculates the second reduction percentage. -/
theorem second_reduction_percentage 
  (first_reduction : ℝ) 
  (final_price_percentage : ℝ) : ℝ :=
let remaining_after_first := 1 - first_reduction
let second_reduction := 1 - (final_price_percentage / remaining_after_first)
second_reduction

/-- Proves that for the given conditions, the second reduction percentage is 23.5%. -/
theorem store_price_reduction : 
  second_reduction_percentage 0.15 0.765 = 0.235 := by
sorry

end second_reduction_percentage_store_price_reduction_l1413_141382


namespace parabola_point_to_directrix_distance_l1413_141391

/-- Represents a parabola opening to the right with equation y² = 2px -/
structure Parabola where
  p : ℝ
  opens_right : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_point_to_directrix_distance
  (C : Parabola) (A : Point)
  (h1 : A.x = 1)
  (h2 : A.y = Real.sqrt 5)
  (h3 : A.y ^ 2 = 2 * C.p * A.x) :
  A.x + C.p / 2 = 9 / 4 := by
  sorry

end parabola_point_to_directrix_distance_l1413_141391


namespace hourly_charge_is_correct_l1413_141362

/-- The hourly charge for renting a bike -/
def hourly_charge : ℝ := 7

/-- The fixed fee for renting a bike -/
def fixed_fee : ℝ := 17

/-- The number of hours Tom rented the bike -/
def rental_hours : ℝ := 9

/-- The total cost Tom paid for renting the bike -/
def total_cost : ℝ := 80

/-- Theorem stating that the hourly charge is correct given the conditions -/
theorem hourly_charge_is_correct : 
  fixed_fee + rental_hours * hourly_charge = total_cost := by sorry

end hourly_charge_is_correct_l1413_141362


namespace ellipse_equation_l1413_141386

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A parabola with equation y² = 4px where p is the focal distance -/
structure Parabola where
  p : ℝ
  h_pos : 0 < p

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos : 0 < r

/-- The theorem stating the conditions and the result to be proved -/
theorem ellipse_equation (e : Ellipse) (p : Parabola) (c : Circle) 
  (h_focus : e.a^2 - e.b^2 = p.p^2) 
  (h_major_axis : 2 * e.a = c.r) 
  (h_parabola : p.p^2 = 3) :
  e.a^2 = 4 ∧ e.b^2 = 1 := by
  sorry

end ellipse_equation_l1413_141386


namespace linear_function_decreasing_l1413_141368

/-- A linear function y = mx + b where m is the slope and b is the y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The linear function y = (k-3)x + 2 -/
def f (k : ℝ) : LinearFunction :=
  { slope := k - 3, intercept := 2 }

/-- A function is decreasing if for any x1 < x2, f(x1) > f(x2) -/
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

/-- The main theorem: The linear function y = (k-3)x + 2 is decreasing iff k < 3 -/
theorem linear_function_decreasing (k : ℝ) :
  isDecreasing (fun x ↦ (f k).slope * x + (f k).intercept) ↔ k < 3 := by
  sorry

end linear_function_decreasing_l1413_141368


namespace max_abs_quadratic_function_bound_l1413_141398

theorem max_abs_quadratic_function_bound (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-1) 1), |f x|
  M ≥ (1/2 : ℝ) := by
sorry

end max_abs_quadratic_function_bound_l1413_141398


namespace fish_rice_trade_l1413_141389

/-- Represents the value of one fish in terms of bags of rice -/
def fish_value (fish bread apple rice : ℚ) : Prop :=
  (5 * fish = 3 * bread) ∧
  (bread = 6 * apple) ∧
  (2 * apple = rice) →
  fish = 9/5 * rice

theorem fish_rice_trade : ∀ (fish bread apple rice : ℚ),
  fish_value fish bread apple rice :=
by
  sorry

end fish_rice_trade_l1413_141389


namespace factorization_equality_l1413_141377

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l1413_141377


namespace regression_slope_effect_l1413_141339

/-- Represents a simple linear regression model -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The predicted value of y given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

/-- The change in y when x increases by one unit -/
def change_in_y (model : LinearRegression) : ℝ :=
  predict model 1 - predict model 0

theorem regression_slope_effect (model : LinearRegression) 
  (h : model = {intercept := 3, slope := -5}) : 
  change_in_y model = -5 := by
  sorry

end regression_slope_effect_l1413_141339


namespace min_value_expression_l1413_141365

theorem min_value_expression (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a < 1) (hb : 0 ≤ b ∧ b < 1) (hc : 0 ≤ c ∧ c < 1) : 
  (1 / ((2 - a) * (2 - b) * (2 - c))) + (1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1/8 := by
  sorry

end min_value_expression_l1413_141365


namespace perpendicular_lines_b_value_l1413_141300

/-- 
If two lines given by the equations 4y + 3x + 6 = 0 and 6y + bx + 5 = 0 are perpendicular,
then b = -8.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 4 * y + 3 * x + 6 = 0 ↔ y = -3/4 * x - 3/2) →
  (∀ x y, 6 * y + b * x + 5 = 0 ↔ y = -b/6 * x - 5/6) →
  ((-3/4) * (-b/6) = -1) →
  b = -8 := by sorry

end perpendicular_lines_b_value_l1413_141300


namespace smallest_cut_prevents_triangle_smallest_cut_is_minimal_l1413_141312

/-- The smallest positive integer that, when subtracted from the original lengths,
    prevents the formation of a triangle. -/
def smallest_cut : ℕ := 2

/-- Original lengths of the sticks -/
def original_lengths : List ℕ := [9, 12, 20]

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The remaining lengths after cutting -/
def remaining_lengths (x : ℕ) : List ℕ :=
  original_lengths.map (λ l => l - x)

theorem smallest_cut_prevents_triangle :
  ∀ x : ℕ, x < smallest_cut →
    ∃ a b c, a::b::c::[] = remaining_lengths x ∧ can_form_triangle a b c :=
by sorry

theorem smallest_cut_is_minimal :
  ¬∃ a b c, a::b::c::[] = remaining_lengths smallest_cut ∧ can_form_triangle a b c :=
by sorry

end smallest_cut_prevents_triangle_smallest_cut_is_minimal_l1413_141312


namespace bob_age_proof_l1413_141302

/-- Bob's age in years -/
def bob_age : ℝ := 51.25

/-- Jim's age in years -/
def jim_age : ℝ := 75 - bob_age

/-- Theorem stating Bob's age given the conditions -/
theorem bob_age_proof :
  (bob_age = 3 * jim_age - 20) ∧
  (bob_age + jim_age = 75) →
  bob_age = 51.25 := by
sorry

end bob_age_proof_l1413_141302


namespace one_common_color_l1413_141395

/-- Given a set of n ≥ 5 colors and n+1 distinct 3-element subsets,
    there exist two subsets that share exactly one element. -/
theorem one_common_color (n : ℕ) (C : Finset ℕ) (A : Fin (n + 1) → Finset ℕ)
  (h_n : n ≥ 5)
  (h_C : C.card = n)
  (h_A_subset : ∀ i, A i ⊆ C)
  (h_A_card : ∀ i, (A i).card = 3)
  (h_A_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card = 1 := by
  sorry

end one_common_color_l1413_141395


namespace equation_solution_l1413_141316

theorem equation_solution (a : ℝ) :
  (a ≠ 0 → ∃! x : ℝ, x ≠ 0 ∧ x ≠ a ∧ 3 * x^2 + 2 * a * x - a^2 = Real.log ((x - a) / (2 * x))) ∧
  (a = 0 → ¬∃ x : ℝ, x ≠ 0 ∧ 3 * x^2 = Real.log (1 / 2)) :=
by sorry

end equation_solution_l1413_141316


namespace balloons_rearrangements_l1413_141313

def word : String := "BALLOONS"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => ¬(is_vowel c))

theorem balloons_rearrangements :
  (vowels.length.factorial / (vowels.countP (· = 'O')).factorial) *
  (consonants.length.factorial / (consonants.countP (· = 'L')).factorial) = 180 := by
  sorry

end balloons_rearrangements_l1413_141313


namespace beth_crayons_l1413_141388

theorem beth_crayons (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 54 → left = 52 → initial = given_away + left → initial = 106 := by
  sorry

end beth_crayons_l1413_141388


namespace inequality_proof_l1413_141369

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
    2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end inequality_proof_l1413_141369


namespace inverse_theorem_not_exists_l1413_141352

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define congruence for triangles
def isCongruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define equality of corresponding angles
def hasEqualAngles (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem inverse_theorem_not_exists :
  ¬(∀ t1 t2 : Triangle, hasEqualAngles t1 t2 → isCongruent t1 t2) :=
sorry

end inverse_theorem_not_exists_l1413_141352


namespace arithmetic_sequence_angles_l1413_141359

theorem arithmetic_sequence_angles (angles : Fin 5 → ℝ) : 
  (∀ i j : Fin 5, i < j → angles i < angles j) →  -- angles are strictly increasing
  (∀ i : Fin 4, angles (i + 1) - angles i = angles (i + 2) - angles (i + 1)) →  -- arithmetic sequence
  angles 0 = 25 →  -- smallest angle
  angles 4 = 105 →  -- largest angle
  ∀ i : Fin 4, angles (i + 1) - angles i = 20 :=
by sorry

end arithmetic_sequence_angles_l1413_141359


namespace triangle_area_triangle_area_is_18_l1413_141306

/-- The area of the triangle formed by the lines y = 8, y = 2 + 2x, and y = 2 - 2x -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (A B C : ℝ × ℝ),
      (A.2 = 8 ∧ B.2 = 2 + 2 * B.1 ∧ C.2 = 2 - 2 * C.1) ∧
      (A.2 = 8 ∧ A.2 = 2 + 2 * A.1) ∧
      (B.2 = 8 ∧ B.2 = 2 - 2 * B.1) ∧
      (C.2 = 2 + 2 * C.1 ∧ C.2 = 2 - 2 * C.1) ∧
      area = 18

/-- The area of the triangle is 18 -/
theorem triangle_area_is_18 : triangle_area 18 := by
  sorry

end triangle_area_triangle_area_is_18_l1413_141306


namespace square_difference_l1413_141367

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) : (x - y)^2 = 16 := by
  sorry

end square_difference_l1413_141367


namespace equality_condition_l1413_141354

theorem equality_condition (a b c k : ℝ) : 
  a + b + c = 1 → (k * (a + b * c) = (a + b) * (a + c) ↔ k = 1) := by
  sorry

end equality_condition_l1413_141354


namespace union_A_B_complement_A_union_B_l1413_141322

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5, 7}

-- Define set B
def B : Set Nat := {3, 5}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by sorry

-- Theorem for (∁ₐA) ∪ B
theorem complement_A_union_B : (U \ A) ∪ B = {2, 3, 4, 5, 6} := by sorry

end union_A_B_complement_A_union_B_l1413_141322


namespace one_thirds_in_nine_fifths_l1413_141390

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 : ℚ) / 3 = 27 / 5 := by sorry

end one_thirds_in_nine_fifths_l1413_141390


namespace solution_set_quadratic_inequality_l1413_141351

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end solution_set_quadratic_inequality_l1413_141351


namespace divisor_of_smallest_six_digit_multiple_l1413_141376

def smallest_six_digit_number : Nat := 100000
def given_number : Nat := 100011
def divisor : Nat := 33337

theorem divisor_of_smallest_six_digit_multiple :
  (given_number = smallest_six_digit_number + 11) →
  (∀ n : Nat, n < given_number → n < smallest_six_digit_number ∨ given_number % n ≠ 0) →
  (given_number % divisor = 0) →
  (given_number / divisor = 3) :=
by sorry

end divisor_of_smallest_six_digit_multiple_l1413_141376


namespace solve_mushroom_problem_l1413_141357

def mushroom_problem (pieces_per_mushroom : ℕ) (total_mushrooms : ℕ) 
  (kenny_pieces : ℕ) (remaining_pieces : ℕ) : Prop :=
  let total_pieces := pieces_per_mushroom * total_mushrooms
  let karla_pieces := total_pieces - (kenny_pieces + remaining_pieces)
  karla_pieces = 42

theorem solve_mushroom_problem :
  mushroom_problem 4 22 38 8 := by sorry

end solve_mushroom_problem_l1413_141357


namespace carol_birthday_invitations_l1413_141307

/-- The number of friends Carol wants to invite -/
def num_friends : ℕ := sorry

/-- The number of invitations in each package -/
def invitations_per_package : ℕ := 3

/-- The number of packages Carol bought -/
def packages_bought : ℕ := 2

/-- The number of extra invitations Carol needs to buy -/
def extra_invitations : ℕ := 3

/-- Theorem stating that the number of friends Carol wants to invite
    is equal to the sum of invitations in bought packs and extra invitations -/
theorem carol_birthday_invitations :
  num_friends = packages_bought * invitations_per_package + extra_invitations := by
  sorry

end carol_birthday_invitations_l1413_141307


namespace w_change_factor_l1413_141348

theorem w_change_factor (w w' m z : ℝ) (h_pos_m : m > 0) (h_pos_z : z > 0) :
  let q := 5 * w / (4 * m * z^2)
  let q' := 5 * w' / (4 * (2 * m) * (3 * z)^2)
  q' = 0.2222222222222222 * q → w' = 4 * w := by
  sorry

end w_change_factor_l1413_141348
