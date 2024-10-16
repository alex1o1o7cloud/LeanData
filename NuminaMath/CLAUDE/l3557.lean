import Mathlib

namespace NUMINAMATH_CALUDE_professor_k_lectures_l3557_355774

def num_jokes : ℕ := 8

theorem professor_k_lectures (num_jokes : ℕ) (h : num_jokes = 8) :
  (Finset.sum (Finset.range 2) (λ i => Nat.choose num_jokes (i + 2))) = 84 := by
  sorry

end NUMINAMATH_CALUDE_professor_k_lectures_l3557_355774


namespace NUMINAMATH_CALUDE_median_could_be_16_l3557_355765

/-- Represents the age distribution of the school band --/
structure AgeDist where
  age13 : Nat
  age14 : Nat
  age15 : Nat
  age16 : Nat

/-- Calculates the total number of members in the band --/
def totalMembers (dist : AgeDist) : Nat :=
  dist.age13 + dist.age14 + dist.age15 + dist.age16

/-- Checks if a given age is the median of the distribution --/
def isMedian (dist : AgeDist) (age : Nat) : Prop :=
  let total := totalMembers dist
  let halfTotal := total / 2
  let countBelow := 
    if age == 13 then 0
    else if age == 14 then dist.age13
    else if age == 15 then dist.age13 + dist.age14
    else dist.age13 + dist.age14 + dist.age15
  countBelow < halfTotal ∧ countBelow + (if age == 16 then dist.age16 else 0) ≥ halfTotal

/-- The main theorem stating that 16 could be the median --/
theorem median_could_be_16 (dist : AgeDist) : 
  dist.age13 = 5 → dist.age14 = 7 → dist.age15 = 13 → ∃ n : Nat, isMedian { age13 := 5, age14 := 7, age15 := 13, age16 := n } 16 :=
sorry

end NUMINAMATH_CALUDE_median_could_be_16_l3557_355765


namespace NUMINAMATH_CALUDE_sequence_negative_start_l3557_355738

def sequence_term (n : ℤ) : ℤ := 21 + 4*n - n^2

theorem sequence_negative_start :
  ∀ n : ℕ, n ≥ 8 → sequence_term n < 0 ∧ 
  ∀ k : ℕ, k < 8 → sequence_term k ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_negative_start_l3557_355738


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l3557_355792

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_positive_r : 0 < r

/-- The statement of the problem -/
theorem ellipse_circle_intersection (e : Ellipse) (c : Circle) :
  e.a = 3 ∧ e.b = 2 ∧
  (∃ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1 ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2) ∧
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁^2 / 9 + y₁^2 / 4 = 1 ∧ (x₁ - c.h)^2 + (y₁ - c.k)^2 = c.r^2) ∧
    (x₂^2 / 9 + y₂^2 / 4 = 1 ∧ (x₂ - c.h)^2 + (y₂ - c.k)^2 = c.r^2) ∧
    (x₃^2 / 9 + y₃^2 / 4 = 1 ∧ (x₃ - c.h)^2 + (y₃ - c.k)^2 = c.r^2) ∧
    (x₄^2 / 9 + y₄^2 / 4 = 1 ∧ (x₄ - c.h)^2 + (y₄ - c.k)^2 = c.r^2) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄)) →
  c.r ≥ Real.sqrt 5 ∧ c.r < 9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l3557_355792


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3557_355797

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3557_355797


namespace NUMINAMATH_CALUDE_sum_division_problem_l3557_355727

/-- Proof of the total amount in a sum division problem -/
theorem sum_division_problem (x y z total : ℚ) : 
  y = 0.45 * x →  -- For each rupee x gets, y gets 45 paisa
  z = 0.5 * x →   -- For each rupee x gets, z gets 50 paisa
  y = 18 →        -- The share of y is Rs. 18
  total = x + y + z →  -- The total is the sum of all shares
  total = 78 := by  -- The total amount is Rs. 78
sorry


end NUMINAMATH_CALUDE_sum_division_problem_l3557_355727


namespace NUMINAMATH_CALUDE_johns_number_l3557_355757

theorem johns_number : ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 64 ∣ n ∧ 45 ∣ n ∧ n = 2880 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_l3557_355757


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3557_355726

structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def is_valid_parallelogram (p : Parallelogram) : Prop :=
  p.A.1 = 2 ∧ p.A.2 = -3 ∧
  p.B.1 = 7 ∧ p.B.2 = 0 ∧
  p.D.1 = -2 ∧ p.D.2 = 5 ∧
  (p.A.1 + p.D.1) / 2 = (p.B.1 + p.C.1) / 2 ∧
  (p.A.2 + p.D.2) / 2 = (p.B.2 + p.C.2) / 2

theorem parallelogram_vertex_sum (p : Parallelogram) 
  (h : is_valid_parallelogram p) : p.C.1 + p.C.2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3557_355726


namespace NUMINAMATH_CALUDE_optimal_planting_strategy_l3557_355775

/-- Represents the cost and planting details for a flower planting project --/
structure FlowerPlanting where
  costA : ℝ  -- Cost per pot of type A flowers
  costB : ℝ  -- Cost per pot of type B flowers
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℝ  -- Survival rate of type A flowers
  survivalRateB : ℝ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Calculates the total cost of planting flowers --/
def totalCost (fp : FlowerPlanting) (potsA : ℕ) : ℝ :=
  fp.costA * potsA + fp.costB * (fp.totalPots - potsA)

/-- Calculates the number of pots to be replaced next year --/
def replacementPots (fp : FlowerPlanting) (potsA : ℕ) : ℝ :=
  (1 - fp.survivalRateA) * potsA + (1 - fp.survivalRateB) * (fp.totalPots - potsA)

/-- Theorem stating the optimal planting strategy and minimum cost --/
theorem optimal_planting_strategy (fp : FlowerPlanting) 
    (h1 : 3 * fp.costA + 4 * fp.costB = 360)
    (h2 : 4 * fp.costA + 3 * fp.costB = 340)
    (h3 : fp.totalPots = 600)
    (h4 : fp.survivalRateA = 0.7)
    (h5 : fp.survivalRateB = 0.9)
    (h6 : fp.maxReplacement = 100) :
    ∃ (optimalA : ℕ), 
      optimalA = 200 ∧ 
      replacementPots fp optimalA ≤ fp.maxReplacement ∧
      ∀ (potsA : ℕ), replacementPots fp potsA ≤ fp.maxReplacement → 
        totalCost fp optimalA ≤ totalCost fp potsA ∧
      totalCost fp optimalA = 32000 := by
  sorry

end NUMINAMATH_CALUDE_optimal_planting_strategy_l3557_355775


namespace NUMINAMATH_CALUDE_abs_neg_five_equals_five_l3557_355798

theorem abs_neg_five_equals_five :
  abs (-5 : ℤ) = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_five_equals_five_l3557_355798


namespace NUMINAMATH_CALUDE_apple_seed_average_l3557_355796

theorem apple_seed_average (total_seeds : ℕ) (pear_avg : ℕ) (grape_avg : ℕ)
  (apple_count : ℕ) (pear_count : ℕ) (grape_count : ℕ) (seeds_needed : ℕ)
  (h1 : total_seeds = 60)
  (h2 : pear_avg = 2)
  (h3 : grape_avg = 3)
  (h4 : apple_count = 4)
  (h5 : pear_count = 3)
  (h6 : grape_count = 9)
  (h7 : seeds_needed = 3) :
  ∃ (apple_avg : ℕ), apple_avg = 6 ∧
    apple_count * apple_avg + pear_count * pear_avg + grape_count * grape_avg
    = total_seeds - seeds_needed :=
by sorry

end NUMINAMATH_CALUDE_apple_seed_average_l3557_355796


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l3557_355783

def is_factor (a n : ℕ) : Prop := n % a = 0

theorem smallest_non_factor_product_of_48 (a b : ℕ) :
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 48 →
  is_factor b 48 →
  ¬ is_factor (a * b) 48 →
  ∀ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ is_factor x 48 ∧ is_factor y 48 ∧ ¬ is_factor (x * y) 48 →
  a * b ≤ x * y →
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l3557_355783


namespace NUMINAMATH_CALUDE_mps_to_kmph_conversion_l3557_355725

-- Define the conversion factor from mps to kmph
def mps_to_kmph_factor : ℝ := 3.6

-- Define the speed in mps
def speed_mps : ℝ := 15

-- Define the speed in kmph
def speed_kmph : ℝ := 54

-- Theorem to prove the conversion
theorem mps_to_kmph_conversion :
  speed_mps * mps_to_kmph_factor = speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_mps_to_kmph_conversion_l3557_355725


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3557_355733

theorem inequality_system_integer_solutions :
  {x : ℤ | 2 * x + 1 > 0 ∧ 2 * x ≤ 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3557_355733


namespace NUMINAMATH_CALUDE_parabola_with_directrix_neg_seven_l3557_355787

/-- Represents a parabola with a vertical axis of symmetry -/
structure Parabola where
  /-- The distance from the vertex to the focus or directrix -/
  p : ℝ
  /-- Indicates whether the parabola opens to the right (true) or left (false) -/
  opensRight : Bool

/-- The standard equation of a parabola -/
def standardEquation (par : Parabola) : ℝ → ℝ → Prop :=
  if par.opensRight then
    fun y x => y^2 = 4 * par.p * x
  else
    fun y x => y^2 = -4 * par.p * x

/-- The equation of the directrix of a parabola -/
def directrixEquation (par : Parabola) : ℝ → Prop :=
  if par.opensRight then
    fun x => x = -par.p
  else
    fun x => x = par.p

theorem parabola_with_directrix_neg_seven (par : Parabola) :
  directrixEquation par = fun x => x = -7 →
  standardEquation par = fun y x => y^2 = 28 * x :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_with_directrix_neg_seven_l3557_355787


namespace NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l3557_355753

/-- The additional time required to fill a cistern with a leak -/
theorem cistern_fill_time_with_leak 
  (normal_fill_time : ℝ) 
  (leak_empty_time : ℝ) 
  (h1 : normal_fill_time = 8) 
  (h2 : leak_empty_time = 40.00000000000001) : 
  (1 / (1 / normal_fill_time - 1 / leak_empty_time)) - normal_fill_time = 2.000000000000003 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l3557_355753


namespace NUMINAMATH_CALUDE_prob_draw_club_is_one_fourth_l3557_355711

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a deck -/
inductive Suit
| Spades | Hearts | Diamonds | Clubs

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The total number of cards in the deck -/
def total_cards : ℕ := 52

/-- The probability of drawing a club from the top of a shuffled deck -/
def prob_draw_club (d : Deck) : ℚ :=
  cards_per_suit / total_cards

theorem prob_draw_club_is_one_fourth (d : Deck) :
  prob_draw_club d = 1 / 4 := by
  sorry

#check prob_draw_club_is_one_fourth

end NUMINAMATH_CALUDE_prob_draw_club_is_one_fourth_l3557_355711


namespace NUMINAMATH_CALUDE_existence_of_x_l3557_355707

/-- A sequence of nonnegative integers satisfying the given condition -/
def SequenceCondition (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
    a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

/-- The main theorem -/
theorem existence_of_x (a : ℕ → ℕ) (h : SequenceCondition a) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a n = ⌊n * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_l3557_355707


namespace NUMINAMATH_CALUDE_total_pigeons_l3557_355728

def initial_pigeons : ℕ := 1
def joined_pigeons : ℕ := 1

theorem total_pigeons : initial_pigeons + joined_pigeons = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_pigeons_l3557_355728


namespace NUMINAMATH_CALUDE_paige_pencils_l3557_355769

theorem paige_pencils (initial_pencils : ℕ) (used_pencils : ℕ) : 
  initial_pencils = 94 → used_pencils = 3 → initial_pencils - used_pencils = 91 := by
  sorry

end NUMINAMATH_CALUDE_paige_pencils_l3557_355769


namespace NUMINAMATH_CALUDE_slope_product_is_two_l3557_355767

/-- Given two lines with slopes m and n, where one line makes twice the angle
    with the horizontal as the other, has 4 times the slope, and is not horizontal,
    prove that the product of their slopes is 2. -/
theorem slope_product_is_two (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 2 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- L₁ makes twice the angle
  m = 4 * n →                                                      -- L₁ has 4 times the slope
  m ≠ 0 →                                                          -- L₁ is not horizontal
  m * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_is_two_l3557_355767


namespace NUMINAMATH_CALUDE_same_solution_value_l3557_355735

theorem same_solution_value (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 5 = 2 ∧ c * x + 4 = 1) ↔ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_value_l3557_355735


namespace NUMINAMATH_CALUDE_hyperbola_parabola_symmetry_l3557_355750

/-- Given a hyperbola and points on a parabola, prove the value of m -/
theorem hyperbola_parabola_symmetry (a b : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ |2*x| = 4) →  -- Condition for hyperbola
  y₁ = a*x₁^2 → y₂ = a*x₂^2 →  -- Points on parabola
  (x₁ + x₂)/2 + m = (y₁ + y₂)/2 →  -- Midpoint on symmetry line
  (y₂ - y₁)/(x₂ - x₁) = -1 →  -- Perpendicular to symmetry line
  x₁*x₂ = -1/2 → 
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_symmetry_l3557_355750


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3557_355784

-- Define the repeating decimal 0.454545...
def repeating_decimal : ℚ := 45 / 99

-- State the theorem
theorem repeating_decimal_equals_fraction : repeating_decimal = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3557_355784


namespace NUMINAMATH_CALUDE_elementary_to_kindergarten_ratio_is_two_to_one_l3557_355766

/-- Represents the purchase of dinosaur models by a school --/
structure ModelPurchase where
  regular_price : ℕ  -- Regular price of each model in dollars
  kindergarten_models : ℕ  -- Number of models for kindergarten library
  total_paid : ℕ  -- Total amount paid in dollars
  discount_percent : ℕ  -- Discount percentage applied

/-- Calculates the ratio of elementary library models to kindergarten library models --/
def elementary_to_kindergarten_ratio (purchase : ModelPurchase) : ℚ :=
  let discounted_price := purchase.regular_price * (100 - purchase.discount_percent) / 100
  let kindergarten_cost := purchase.kindergarten_models * purchase.regular_price
  let elementary_cost := purchase.total_paid - kindergarten_cost
  let elementary_models := elementary_cost / discounted_price
  elementary_models / purchase.kindergarten_models

/-- Theorem stating the ratio of elementary to kindergarten models is 2:1 --/
theorem elementary_to_kindergarten_ratio_is_two_to_one 
  (purchase : ModelPurchase)
  (h1 : purchase.regular_price = 100)
  (h2 : purchase.kindergarten_models = 2)
  (h3 : purchase.total_paid = 570)
  (h4 : purchase.discount_percent = 5)
  (h5 : purchase.kindergarten_models + 
        (purchase.total_paid - purchase.kindergarten_models * purchase.regular_price) / 
        (purchase.regular_price * (100 - purchase.discount_percent) / 100) > 5) :
  elementary_to_kindergarten_ratio purchase = 2 := by
  sorry

end NUMINAMATH_CALUDE_elementary_to_kindergarten_ratio_is_two_to_one_l3557_355766


namespace NUMINAMATH_CALUDE_rectangle_perimeter_increase_l3557_355712

theorem rectangle_perimeter_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let initial_perimeter := 2 * (l + w)
  let new_perimeter := 2 * (1.1 * l + 1.1 * w)
  new_perimeter / initial_perimeter = 1.1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_increase_l3557_355712


namespace NUMINAMATH_CALUDE_distance_between_points_l3557_355721

theorem distance_between_points (x : ℝ) : 
  |(3 + x) - (3 - x)| = 8 → |x| = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3557_355721


namespace NUMINAMATH_CALUDE_carson_clawed_39_times_l3557_355723

/-- The number of times Carson gets clawed in the zoo enclosure. -/
def total_claws (num_wombats : ℕ) (num_rheas : ℕ) (claws_per_wombat : ℕ) (claws_per_rhea : ℕ) : ℕ :=
  num_wombats * claws_per_wombat + num_rheas * claws_per_rhea

/-- Theorem stating that Carson gets clawed 39 times. -/
theorem carson_clawed_39_times :
  total_claws 9 3 4 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_carson_clawed_39_times_l3557_355723


namespace NUMINAMATH_CALUDE_basketball_practice_time_l3557_355795

theorem basketball_practice_time (school_day_practice : ℕ) : 
  (5 * school_day_practice + 2 * (2 * school_day_practice) = 135) → 
  school_day_practice = 15 := by
sorry

end NUMINAMATH_CALUDE_basketball_practice_time_l3557_355795


namespace NUMINAMATH_CALUDE_remainder_problem_l3557_355745

theorem remainder_problem (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3557_355745


namespace NUMINAMATH_CALUDE_journey_speed_l3557_355743

/-- Proves that given a journey where 75% is traveled at 50 mph and 25% at S mph,
    if the average speed for the entire journey is 50 mph, then S must equal 50 mph. -/
theorem journey_speed (D : ℝ) (S : ℝ) (h1 : D > 0) :
  (D / ((0.75 * D / 50) + (0.25 * D / S)) = 50) → S = 50 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l3557_355743


namespace NUMINAMATH_CALUDE_fraction_enlargement_l3557_355781

theorem fraction_enlargement (x y : ℝ) (h : 3 * x - y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - (3 * y)) = 3 * ((2 * x * y) / (3 * x - y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l3557_355781


namespace NUMINAMATH_CALUDE_cone_base_radius_l3557_355708

/-- Represents a cone with given properties -/
structure Cone where
  surface_area : ℝ
  lateral_surface_semicircle : Prop

/-- Theorem: Given a cone with surface area 12π and lateral surface unfolding into a semicircle, 
    the radius of its base circle is 2 -/
theorem cone_base_radius 
  (cone : Cone) 
  (h1 : cone.surface_area = 12 * Real.pi) 
  (h2 : cone.lateral_surface_semicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r > 0 ∧ 
  cone.surface_area = Real.pi * r^2 + Real.pi * r * (2 * r) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3557_355708


namespace NUMINAMATH_CALUDE_even_function_solution_set_l3557_355768

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is symmetric about the origin -/
def HasSymmetricDomain (f : ℝ → ℝ) (a : ℝ) : Prop :=
  Set.Icc (1 + a) 1 = Set.Icc (-1) 1

theorem even_function_solution_set
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : f = fun x ↦ a * x^2 + b * x + 2)
  (h2 : IsEven f)
  (h3 : HasSymmetricDomain f a) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_solution_set_l3557_355768


namespace NUMINAMATH_CALUDE_max_sum_abc_l3557_355734

def An (a n : ℕ) : ℕ := a * (8^n - 1) / 7
def Bn (b n : ℕ) : ℕ := b * (6^n - 1) / 5
def Cn (c n : ℕ) : ℕ := c * (10^(3*n) - 1) / 9

theorem max_sum_abc :
  ∃ (a b c : ℕ),
    (0 < a ∧ a ≤ 9) ∧
    (0 < b ∧ b ≤ 9) ∧
    (0 < c ∧ c ≤ 9) ∧
    (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ 
      Cn c n₁ - Bn b n₁ = (An a n₁)^3 ∧
      Cn c n₂ - Bn b n₂ = (An a n₂)^3) ∧
    (∀ (a' b' c' : ℕ),
      (0 < a' ∧ a' ≤ 9) →
      (0 < b' ∧ b' ≤ 9) →
      (0 < c' ∧ c' ≤ 9) →
      (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ 
        Cn c' n₁ - Bn b' n₁ = (An a' n₁)^3 ∧
        Cn c' n₂ - Bn b' n₂ = (An a' n₂)^3) →
      a + b + c ≥ a' + b' + c') ∧
    a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_abc_l3557_355734


namespace NUMINAMATH_CALUDE_model_building_height_l3557_355718

/-- Proves that the height of the model building is 1.2 meters given the conditions of the problem -/
theorem model_building_height
  (actual_tower_height : ℝ)
  (actual_tower_volume : ℝ)
  (actual_building_height : ℝ)
  (model_tower_volume : ℝ)
  (h1 : actual_tower_height = 60)
  (h2 : actual_tower_volume = 200000)
  (h3 : actual_building_height = 120)
  (h4 : model_tower_volume = 0.2)
  : ℝ :=
by
  sorry

#check model_building_height

end NUMINAMATH_CALUDE_model_building_height_l3557_355718


namespace NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l3557_355740

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid := Position → ℕ

/-- Creates a spiral grid with the given dimensions -/
def create_spiral_grid (n : ℕ) : SpiralGrid :=
  sorry

/-- Returns the numbers in a given row of the grid -/
def numbers_in_row (grid : SpiralGrid) (row : ℕ) : List ℕ :=
  sorry

theorem spiral_grid_third_row_sum :
  let grid := create_spiral_grid 17
  let third_row_numbers := numbers_in_row grid 3
  let min_number := third_row_numbers.minimum?
  let max_number := third_row_numbers.maximum?
  ∀ min max, min_number = some min → max_number = some max →
    min + max = 577 := by
  sorry

end NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l3557_355740


namespace NUMINAMATH_CALUDE_men_in_room_l3557_355730

/-- Given the initial ratio of men to women is 4:5, prove that after 2 men enter, 3 women leave, 
    and the number of women doubles to 24, the number of men in the room is 14. -/
theorem men_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  2 * (initial_women - 3) = 24 →
  initial_men + 2 = 14 := by
  sorry

#check men_in_room

end NUMINAMATH_CALUDE_men_in_room_l3557_355730


namespace NUMINAMATH_CALUDE_line_point_at_t_4_l3557_355709

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point_at : ℝ → ℝ × ℝ × ℝ

/-- Given a parameterized line with known points at t = 1 and t = -1, 
    prove that the point at t = 4 is (-27, 57, 27) -/
theorem line_point_at_t_4 
  (line : ParameterizedLine)
  (h1 : line.point_at 1 = (-3, 9, 12))
  (h2 : line.point_at (-1) = (4, -4, 2)) :
  line.point_at 4 = (-27, 57, 27) := by
sorry


end NUMINAMATH_CALUDE_line_point_at_t_4_l3557_355709


namespace NUMINAMATH_CALUDE_jacob_age_l3557_355785

/-- Given Rehana's current age, her age relative to Phoebe's in 5 years, and Jacob's age relative to Phoebe's, prove Jacob's current age. -/
theorem jacob_age (rehana_age : ℕ) (phoebe_age : ℕ) (jacob_age : ℕ) : 
  rehana_age = 25 →
  rehana_age + 5 = 3 * (phoebe_age + 5) →
  jacob_age = 3 * phoebe_age / 5 →
  jacob_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_jacob_age_l3557_355785


namespace NUMINAMATH_CALUDE_yangmei_sales_l3557_355702

/-- Yangmei sales problem -/
theorem yangmei_sales (total_weight : ℕ) (round_weight round_price square_weight square_price : ℕ) 
  (h_total : total_weight = 1000)
  (h_round : round_weight = 8 ∧ round_price = 160)
  (h_square : square_weight = 18 ∧ square_price = 270) :
  (∃ a : ℕ, a * round_price + a * square_price = 8600 → a = 20) ∧
  (∃ x y : ℕ, x * round_price + y * square_price = 16760 ∧ 
              x * round_weight + y * square_weight = total_weight →
              x = 44 ∧ y = 36) ∧
  (∃ b : ℕ, b > 0 ∧ 
            (∃ m n : ℕ, (m + b) * round_weight + n * square_weight = total_weight ∧
                        m * round_price + n * square_price = 16760) →
            b = 9 ∨ b = 18) := by
  sorry

end NUMINAMATH_CALUDE_yangmei_sales_l3557_355702


namespace NUMINAMATH_CALUDE_min_value_theorem_l3557_355746

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a) + (3 / b) ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀) + (3 / b₀) = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3557_355746


namespace NUMINAMATH_CALUDE_rice_mixture_price_l3557_355700

theorem rice_mixture_price (price1 price2 proportion1 proportion2 : ℚ) 
  (h1 : price1 = 31/10)
  (h2 : price2 = 36/10)
  (h3 : proportion1 = 7)
  (h4 : proportion2 = 3)
  : (price1 * proportion1 + price2 * proportion2) / (proportion1 + proportion2) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l3557_355700


namespace NUMINAMATH_CALUDE_bowtie_problem_l3557_355755

-- Define the bowtie operation
noncomputable def bowtie (c d : ℝ) : ℝ := c + 1 + Real.sqrt (d + Real.sqrt (d + Real.sqrt d))

-- State the theorem
theorem bowtie_problem (h : ℝ) (hyp : bowtie 8 h = 12) : h = 6 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_problem_l3557_355755


namespace NUMINAMATH_CALUDE_triangle_rds_area_l3557_355729

/-- The area of a triangle RDS with given coordinates and perpendicular sides -/
theorem triangle_rds_area (k : ℝ) : 
  let R : ℝ × ℝ := (0, 15)
  let D : ℝ × ℝ := (3, 15)
  let S : ℝ × ℝ := (0, k)
  -- RD is perpendicular to RS (implied by coordinates)
  (45 - 3 * k) / 2 = (1 / 2) * 3 * (15 - k) := by sorry

end NUMINAMATH_CALUDE_triangle_rds_area_l3557_355729


namespace NUMINAMATH_CALUDE_estimate_total_children_l3557_355701

theorem estimate_total_children (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0) (h4 : n ≤ m) (h5 : n ≤ k) :
  ∃ (total : ℚ), total = k * (m / n) :=
sorry

end NUMINAMATH_CALUDE_estimate_total_children_l3557_355701


namespace NUMINAMATH_CALUDE_local_maximum_at_one_l3557_355778

/-- The function y = (x+1)/(x^2+3) has a local maximum at x = 1 -/
theorem local_maximum_at_one :
  ∃ δ > 0, ∀ x : ℝ, x ≠ 1 → |x - 1| < δ →
    (x + 1) / (x^2 + 3) ≤ (1 + 1) / (1^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_local_maximum_at_one_l3557_355778


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3557_355741

theorem no_integer_solutions : ¬∃ (a b : ℤ), (1 : ℚ) / a + (1 : ℚ) / b = -(1 : ℚ) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3557_355741


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3557_355731

theorem imaginary_part_of_z (z : ℂ) (h : 1 + z * Complex.I = z - 2 * Complex.I) :
  z.im = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3557_355731


namespace NUMINAMATH_CALUDE_quadratic_function_properties_range_of_g_l3557_355705

-- Define the quadratic function f
def f (x : ℝ) : ℝ := -2 * x^2 - 4 * x

-- State the theorem
theorem quadratic_function_properties :
  -- The vertex of f is (-1, 2)
  (f (-1) = 2 ∧ ∀ x, f x ≤ f (-1)) ∧
  -- f passes through the origin
  f 0 = 0 ∧
  -- The range of f(2x) is (-∞, 0)
  (∀ y, (∃ x, f (2*x) = y) ↔ y < 0) := by
sorry

-- Define g as f(2x)
def g (x : ℝ) : ℝ := f (2*x)

-- Additional theorem for the range of g
theorem range_of_g :
  (∀ y, (∃ x, g x = y) ↔ y < 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_range_of_g_l3557_355705


namespace NUMINAMATH_CALUDE_number_of_players_is_64_l3557_355771

/-- The cost of a pair of shoes in dollars -/
def shoe_cost : ℕ := 12

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := shoe_cost + 8

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := jersey_cost / 2

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (shoe_cost + jersey_cost) + cap_cost

/-- The total expenses for all players' equipment in dollars -/
def total_expenses : ℕ := 4760

theorem number_of_players_is_64 : 
  ∃ n : ℕ, n * player_cost = total_expenses ∧ n = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_players_is_64_l3557_355771


namespace NUMINAMATH_CALUDE_problem_solution_l3557_355748

theorem problem_solution : 
  (5 * Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2) ∧
  ((2 * Real.sqrt 3 - 1)^2 + Real.sqrt 24 / Real.sqrt 2 = 13 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3557_355748


namespace NUMINAMATH_CALUDE_johns_bonus_is_twenty_l3557_355786

/-- Calculate the performance bonus for John's job --/
def performance_bonus (normal_wage : ℝ) (normal_hours : ℝ) (extra_hours : ℝ) (bonus_rate : ℝ) : ℝ :=
  (normal_hours + extra_hours) * bonus_rate - normal_wage

/-- Theorem stating that John's performance bonus is $20 per day --/
theorem johns_bonus_is_twenty :
  performance_bonus 80 8 2 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_bonus_is_twenty_l3557_355786


namespace NUMINAMATH_CALUDE_unique_element_in_S_l3557_355742

-- Define the set
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

-- Theorem statement
theorem unique_element_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end NUMINAMATH_CALUDE_unique_element_in_S_l3557_355742


namespace NUMINAMATH_CALUDE_additional_people_needed_l3557_355770

/-- The number of person-hours required to mow the lawn -/
def personHours : ℕ := 32

/-- The initial number of people who can mow the lawn in 8 hours -/
def initialPeople : ℕ := 4

/-- The desired time to mow the lawn -/
def desiredTime : ℕ := 3

/-- The total number of people needed to mow the lawn in the desired time -/
def totalPeopleNeeded : ℕ := (personHours + desiredTime - 1) / desiredTime

theorem additional_people_needed :
  totalPeopleNeeded - initialPeople = 7 :=
by sorry

end NUMINAMATH_CALUDE_additional_people_needed_l3557_355770


namespace NUMINAMATH_CALUDE_set_equation_solution_l3557_355756

theorem set_equation_solution (p a b : ℝ) : 
  let A := {x : ℝ | x^2 - p*x + 15 = 0}
  let B := {x : ℝ | x^2 - a*x - b = 0}
  (A ∪ B = {2, 3, 5} ∧ A ∩ B = {3}) → (p = 8 ∧ a = 5 ∧ b = -6) := by
  sorry

end NUMINAMATH_CALUDE_set_equation_solution_l3557_355756


namespace NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l3557_355737

theorem probability_second_genuine_given_first_genuine 
  (total_products : ℕ) 
  (genuine_products : ℕ) 
  (defective_products : ℕ) 
  (h1 : total_products = genuine_products + defective_products)
  (h2 : genuine_products = 6)
  (h3 : defective_products = 4) :
  (genuine_products - 1) / (total_products - 1) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l3557_355737


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3557_355758

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 340 →
  final_price = 231.2 →
  second_discount = 0.15 →
  ∃ (first_discount : ℝ),
    first_discount = 0.2 ∧
    final_price = original_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3557_355758


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l3557_355720

theorem meaningful_expression_range (m : ℝ) : 
  (∃ (x : ℝ), x = (m - 1).sqrt / (m - 2)) ↔ (m ≥ 1 ∧ m ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l3557_355720


namespace NUMINAMATH_CALUDE_coupon1_best_discount_best_prices_l3557_355788

/-- Represents the discount offered by Coupon 1 -/
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

/-- Represents the discount offered by Coupon 2 -/
def coupon2_discount : ℝ := 50

/-- Represents the discount offered by Coupon 3 -/
def coupon3_discount (x : ℝ) : ℝ := 0.25 * (x - 250)

/-- Theorem stating when Coupon 1 offers the best discount -/
theorem coupon1_best_discount (x : ℝ) :
  (x ≥ 200 ∧ x ≥ 250) →
  (coupon1_discount x > coupon2_discount ∧
   coupon1_discount x > coupon3_discount x) ↔
  (333.33 < x ∧ x < 625) :=
by sorry

/-- Checks if a given price satisfies the condition for Coupon 1 being the best -/
def is_coupon1_best (price : ℝ) : Prop :=
  333.33 < price ∧ price < 625

/-- Theorem stating which of the given prices satisfy the condition -/
theorem best_prices :
  is_coupon1_best 349.95 ∧
  is_coupon1_best 399.95 ∧
  is_coupon1_best 449.95 ∧
  is_coupon1_best 499.95 ∧
  ¬is_coupon1_best 299.95 :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_discount_best_prices_l3557_355788


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l3557_355717

def numbers : List ℝ := [17, 25, 38]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l3557_355717


namespace NUMINAMATH_CALUDE_race_completion_time_l3557_355780

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- A beats B by 40 meters
  960 = b.speed * a.time ∧
  -- A beats B by 10 seconds
  b.time = a.time + 10

/-- The theorem stating A's completion time -/
theorem race_completion_time (a b : Runner) (h : Race a b) : a.time = 250 := by
  sorry

end NUMINAMATH_CALUDE_race_completion_time_l3557_355780


namespace NUMINAMATH_CALUDE_gold_bar_weight_l3557_355760

/-- Proves that in an arithmetic sequence of 5 terms where the first term is 4 
    and the last term is 2, the second term is 7/2. -/
theorem gold_bar_weight (a : Fin 5 → ℚ) 
  (h_arith : ∀ i j : Fin 5, a j - a i = (j - i : ℚ) * (a 1 - a 0))
  (h_first : a 0 = 4)
  (h_last : a 4 = 2) : 
  a 1 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_gold_bar_weight_l3557_355760


namespace NUMINAMATH_CALUDE_auction_result_l3557_355793

/-- Calculates the total amount received from selling a TV and a phone at an auction -/
def auction_total (tv_cost phone_cost : ℚ) (tv_increase phone_increase : ℚ) : ℚ :=
  (tv_cost + tv_cost * tv_increase) + (phone_cost + phone_cost * phone_increase)

/-- Theorem stating the total amount received from the auction -/
theorem auction_result : 
  auction_total 500 400 (2/5) (40/100) = 1260 := by sorry

end NUMINAMATH_CALUDE_auction_result_l3557_355793


namespace NUMINAMATH_CALUDE_f_composition_value_l3557_355751

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then (1/2)^x
  else if 0 < x ∧ x < 1 then Real.log x / Real.log 4
  else 0  -- This case is added to make the function total

-- State the theorem
theorem f_composition_value : f (f 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3557_355751


namespace NUMINAMATH_CALUDE_shelby_poster_purchase_l3557_355704

/-- Calculates the number of posters Shelby can buy given the problem conditions --/
def calculate_posters (initial_amount coupon_value tax_rate : ℚ)
  (book1_cost book2_cost bookmark_cost pencils_cost notebook_cost poster_cost : ℚ)
  (discount_rate1 discount_rate2 : ℚ)
  (discount_threshold1 discount_threshold2 : ℚ) : ℕ :=
  sorry

/-- Theorem stating that Shelby can buy exactly 4 posters --/
theorem shelby_poster_purchase :
  let initial_amount : ℚ := 60
  let book1_cost : ℚ := 15
  let book2_cost : ℚ := 9
  let bookmark_cost : ℚ := 3.5
  let pencils_cost : ℚ := 4.8
  let notebook_cost : ℚ := 6.2
  let poster_cost : ℚ := 6
  let discount_rate1 : ℚ := 0.15
  let discount_rate2 : ℚ := 0.10
  let discount_threshold1 : ℚ := 40
  let discount_threshold2 : ℚ := 25
  let coupon_value : ℚ := 5
  let tax_rate : ℚ := 0.08
  calculate_posters initial_amount coupon_value tax_rate
    book1_cost book2_cost bookmark_cost pencils_cost notebook_cost poster_cost
    discount_rate1 discount_rate2 discount_threshold1 discount_threshold2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_shelby_poster_purchase_l3557_355704


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3557_355777

/-- Given vectors a and b, prove that a is perpendicular to b iff n = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) :
  a = (3, 2) → b.1 = 2 → a.1 * b.1 + a.2 * b.2 = 0 ↔ b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3557_355777


namespace NUMINAMATH_CALUDE_simplify_expression_l3557_355773

theorem simplify_expression :
  let x : ℝ := 3
  let expr := (Real.sqrt (x - 2 * Real.sqrt 2)) / (Real.sqrt (x^2 - 4*x*Real.sqrt 2 + 8)) -
               (Real.sqrt (x + 2 * Real.sqrt 2)) / (Real.sqrt (x^2 + 4*x*Real.sqrt 2 + 8))
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3557_355773


namespace NUMINAMATH_CALUDE_eesha_travel_time_l3557_355776

/-- Eesha's usual time to reach her office -/
def usual_time : ℝ := 60

/-- The additional time taken when driving slower -/
def additional_time : ℝ := 20

/-- The ratio of slower speed to usual speed -/
def speed_ratio : ℝ := 0.75

theorem eesha_travel_time :
  usual_time = 60 ∧
  additional_time = usual_time / speed_ratio - usual_time :=
by sorry

end NUMINAMATH_CALUDE_eesha_travel_time_l3557_355776


namespace NUMINAMATH_CALUDE_boys_ratio_in_class_l3557_355772

theorem boys_ratio_in_class (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (n : ℚ) / (n + m : ℚ) = 2 / 5 ↔ 
  (n : ℚ) / (n + m : ℚ) = 2 / 3 * (m : ℚ) / (n + m : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_boys_ratio_in_class_l3557_355772


namespace NUMINAMATH_CALUDE_desired_depth_is_18_l3557_355724

/-- Represents the digging scenario with given parameters -/
structure DiggingScenario where
  initial_men : ℕ
  initial_hours : ℕ
  initial_depth : ℕ
  new_hours : ℕ
  extra_men : ℕ

/-- Calculates the desired depth for a given digging scenario -/
def desired_depth (scenario : DiggingScenario) : ℚ :=
  (scenario.initial_men * scenario.initial_hours * scenario.initial_depth : ℚ) /
  ((scenario.initial_men + scenario.extra_men) * scenario.new_hours)

/-- Theorem stating that the desired depth for the given scenario is 18 meters -/
theorem desired_depth_is_18 (scenario : DiggingScenario) 
    (h1 : scenario.initial_men = 9)
    (h2 : scenario.initial_hours = 8)
    (h3 : scenario.initial_depth = 30)
    (h4 : scenario.new_hours = 6)
    (h5 : scenario.extra_men = 11) :
  desired_depth scenario = 18 := by
  sorry

#eval desired_depth { initial_men := 9, initial_hours := 8, initial_depth := 30, new_hours := 6, extra_men := 11 }

end NUMINAMATH_CALUDE_desired_depth_is_18_l3557_355724


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l3557_355736

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a row contains 2, 3, and 4 -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ({2, 3, 4} : Finset Nat) = {g row 0, g row 1, g row 2}

/-- Checks if a column contains 2, 3, and 4 -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ({2, 3, 4} : Finset Nat) = {g 0 col, g 1 col, g 2 col}

/-- Checks if the grid satisfies all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  g 0 0 = 2 ∧
  g 1 1 = 3

theorem sum_of_A_and_B (g : Grid) (h : valid_grid g) : g 2 0 + g 0 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l3557_355736


namespace NUMINAMATH_CALUDE_prob_white_same_color_five_balls_l3557_355764

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of drawing two white balls given that the drawn balls are of the same color -/
def prob_white_given_same_color (total white black : ℕ) : ℚ :=
  let white_ways := choose white 2
  let black_ways := choose black 2
  white_ways / (white_ways + black_ways)

theorem prob_white_same_color_five_balls :
  prob_white_given_same_color 5 3 2 = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_white_same_color_five_balls_l3557_355764


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3557_355789

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3557_355789


namespace NUMINAMATH_CALUDE_three_digit_subtraction_result_zero_l3557_355710

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Subtracts the sum of digits from a number -/
def subtract_sum_of_digits (n : ℕ) : ℕ :=
  n - sum_of_digits n

/-- Applies the subtraction process n times -/
def apply_n_times (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => apply_n_times n (subtract_sum_of_digits x)

/-- The main theorem to be proved -/
theorem three_digit_subtraction_result_zero (x : ThreeDigitNumber) :
  apply_n_times 100 x.value = 0 :=
sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_result_zero_l3557_355710


namespace NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l3557_355749

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x + 3| = q) (h2 : x > -3) :
  x + q = 2*q - 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l3557_355749


namespace NUMINAMATH_CALUDE_farm_animals_l3557_355761

/-- Given a farm with hens and cows, prove the number of hens -/
theorem farm_animals (total_heads : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) :
  total_heads = 44 →
  total_feet = 140 →
  total_heads = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 18 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3557_355761


namespace NUMINAMATH_CALUDE_power_greater_than_linear_l3557_355794

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_linear_l3557_355794


namespace NUMINAMATH_CALUDE_english_marks_calculation_l3557_355791

def average_marks : ℝ := 70
def num_subjects : ℕ := 5
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

theorem english_marks_calculation :
  ∃ (english_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℝ) / num_subjects = average_marks ∧
    english_marks = 51 := by
  sorry

end NUMINAMATH_CALUDE_english_marks_calculation_l3557_355791


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3557_355716

theorem quadratic_roots_sum_of_squares (m : ℝ) 
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ x₂^2 - m*x₂ + 2*m - 1 = 0)
  (h2 : ∃ x₁ x₂ : ℝ, x₁^2 + x₂^2 = 23 ∧ x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ x₂^2 - m*x₂ + 2*m - 1 = 0) :
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3557_355716


namespace NUMINAMATH_CALUDE_number_of_women_at_event_l3557_355759

/-- Proves that the number of women at an event is 20, given the specified dancing conditions -/
theorem number_of_women_at_event (num_men : ℕ) (men_dance_count : ℕ) (women_dance_count : ℕ) 
  (h1 : num_men = 15)
  (h2 : men_dance_count = 4)
  (h3 : women_dance_count = 3)
  : (num_men * men_dance_count) / women_dance_count = 20 := by
  sorry

#check number_of_women_at_event

end NUMINAMATH_CALUDE_number_of_women_at_event_l3557_355759


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3557_355706

theorem digit_sum_problem (a b c x s z : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → x ≠ 0 → s ≠ 0 → z ≠ 0 →
  a + b = x →
  x + c = s →
  s + a = z →
  b + c + z = 16 →
  s = 8 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3557_355706


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3557_355722

theorem quadratic_solution_difference (x : ℝ) : 
  (x^2 - 5*x + 12 = 2*x + 60) → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1^2 - 5*x1 + 12 = 2*x1 + 60) ∧ 
    (x2^2 - 5*x2 + 12 = 2*x2 + 60) ∧ 
    |x1 - x2| = Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3557_355722


namespace NUMINAMATH_CALUDE_f_period_f_definition_f_negative_one_l3557_355779

def f (x : ℝ) : ℝ := sorry

theorem f_period (x : ℝ) : f (x + 2) = f x := sorry

theorem f_definition (x : ℝ) (h : x ∈ Set.Icc 1 3) : f x = x - 2 := sorry

theorem f_negative_one : f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_f_period_f_definition_f_negative_one_l3557_355779


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_l3557_355732

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | |x| < 5}
def T : Set ℝ := {x : ℝ | x^2 + 4*x - 21 < 0}

-- State the theorem
theorem S_intersect_T_eq : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_l3557_355732


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3557_355714

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, 100 ≤ n ∧ n < 104 → ¬(n % 8 = 0)) ∧ 
  104 % 8 = 0 ∧ 
  104 ≥ 100 ∧ 
  104 < 1000 := by
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3557_355714


namespace NUMINAMATH_CALUDE_line_projection_onto_plane_line_projection_onto_plane_ratio_form_l3557_355744

/-- Given a line in 3D space defined by two equations and a plane, 
    this theorem states the equation of the projection of the line onto the plane. -/
theorem line_projection_onto_plane :
  ∀ (x y z : ℝ),
  (3*x - 2*y - z + 4 = 0 ∧ x - 4*y - 3*z - 2 = 0) →  -- Line equations
  (5*x + 2*y + 2*z - 7 = 0) →                        -- Plane equation
  ∃ (t : ℝ), 
    x = -2*t + 1 ∧                                   -- Parametric form of
    y = -14*t + 1 ∧                                  -- the projected line
    z = 19*t :=
by sorry

/-- An alternative formulation of the projection theorem using ratios. -/
theorem line_projection_onto_plane_ratio_form :
  ∀ (x y z : ℝ),
  (3*x - 2*y - z + 4 = 0 ∧ x - 4*y - 3*z - 2 = 0) →  -- Line equations
  (5*x + 2*y + 2*z - 7 = 0) →                        -- Plane equation
  (x - 1) / (-2) = (y - 1) / (-14) ∧ (x - 1) / (-2) = z / 19 :=
by sorry

end NUMINAMATH_CALUDE_line_projection_onto_plane_line_projection_onto_plane_ratio_form_l3557_355744


namespace NUMINAMATH_CALUDE_kimberly_skittles_l3557_355799

/-- Calculates the total number of Skittles Kimberly has -/
def total_skittles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Proves that Kimberly has 12 Skittles in total -/
theorem kimberly_skittles : total_skittles 5 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l3557_355799


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l3557_355715

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l3557_355715


namespace NUMINAMATH_CALUDE_sqrt_floor_equality_l3557_355747

theorem sqrt_floor_equality (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_floor_equality_l3557_355747


namespace NUMINAMATH_CALUDE_jean_calories_consumed_l3557_355754

/-- Calculates the total calories consumed by Jean while writing her paper. -/
def total_calories (pages : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  let donuts := (pages + pages_per_donut - 1) / pages_per_donut
  donuts * calories_per_donut

/-- Proves that Jean consumes 1260 calories while writing her paper. -/
theorem jean_calories_consumed :
  total_calories 20 3 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_jean_calories_consumed_l3557_355754


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_two_to_one_l3557_355703

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 3 -/
structure Square where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Represents the configuration of points and lines in the square -/
structure SquareConfiguration where
  square : Square
  t : Point
  u : Point
  v : Point
  w : Point

/-- The ratio of shaded to unshaded area in the square configuration -/
def shadedToUnshadedRatio (config : SquareConfiguration) : ℚ := 2

/-- Theorem stating that the ratio of shaded to unshaded area is 2:1 -/
theorem shaded_to_unshaded_ratio_is_two_to_one (config : SquareConfiguration) :
  shadedToUnshadedRatio config = 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_two_to_one_l3557_355703


namespace NUMINAMATH_CALUDE_expand_expression_l3557_355790

theorem expand_expression (y : ℝ) : (7 * y + 12) * (3 * y) = 21 * y^2 + 36 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3557_355790


namespace NUMINAMATH_CALUDE_angle_2_measure_l3557_355739

-- Define complementary angles
def complementary (a1 a2 : ℝ) : Prop := a1 + a2 = 90

-- Theorem statement
theorem angle_2_measure (angle1 angle2 : ℝ) 
  (h1 : complementary angle1 angle2) (h2 : angle1 = 25) : 
  angle2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_angle_2_measure_l3557_355739


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3557_355719

/-- Proves that the cost of an adult ticket is $12 given the theater's seating and sales information --/
theorem adult_ticket_cost (total_seats : ℕ) (child_ticket_price : ℕ) (total_revenue : ℕ) (child_tickets_sold : ℕ) : 
  total_seats = 80 →
  child_ticket_price = 5 →
  total_revenue = 519 →
  child_tickets_sold = 63 →
  (total_seats - child_tickets_sold) * 12 + child_tickets_sold * child_ticket_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3557_355719


namespace NUMINAMATH_CALUDE_ralph_cards_l3557_355762

/-- The number of cards Ralph has after various changes. -/
def final_cards (initial : ℕ) (from_father : ℕ) (from_sister : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_father + from_sister - traded - lost

/-- Theorem stating that Ralph ends up with 12 cards given the specific card changes. -/
theorem ralph_cards : final_cards 4 8 5 3 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ralph_cards_l3557_355762


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l3557_355713

def original_cost : ℝ := 1200
def discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.08

def total_spent : ℝ :=
  let discounted_cost := original_cost * (1 - discount_rate)
  let other_toys_with_tax := discounted_cost * (1 + tax_rate)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_with_tax := lightsaber_cost * (1 + tax_rate)
  other_toys_with_tax + lightsaber_with_tax

theorem total_spent_is_correct :
  total_spent = 3628.80 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l3557_355713


namespace NUMINAMATH_CALUDE_variance_describes_dispersion_l3557_355763

-- Define the type for statistical measures
inductive StatMeasure
  | Mean
  | Variance
  | Median
  | Mode

-- Define a property for measures that describe dispersion
def describes_dispersion (m : StatMeasure) : Prop :=
  match m with
  | StatMeasure.Variance => True
  | _ => False

-- Theorem statement
theorem variance_describes_dispersion :
  ∀ m : StatMeasure, describes_dispersion m ↔ m = StatMeasure.Variance :=
by sorry

end NUMINAMATH_CALUDE_variance_describes_dispersion_l3557_355763


namespace NUMINAMATH_CALUDE_smallest_sum_xyz_l3557_355782

theorem smallest_sum_xyz (x y z : ℕ+) 
  (eq1 : (x.val + y.val) * (y.val + z.val) = 2016)
  (eq2 : (x.val + y.val) * (z.val + x.val) = 1080) :
  (∀ a b c : ℕ+, 
    (a.val + b.val) * (b.val + c.val) = 2016 → 
    (a.val + b.val) * (c.val + a.val) = 1080 → 
    x.val + y.val + z.val ≤ a.val + b.val + c.val) ∧
  x.val + y.val + z.val = 61 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_xyz_l3557_355782


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l3557_355752

/-- Given an arithmetic sequence with first term 3 and common difference 4,
    the 20th term of the sequence is 79. -/
theorem arithmetic_sequence_20th_term :
  let a₁ : ℕ := 3  -- first term
  let d : ℕ := 4   -- common difference
  let n : ℕ := 20  -- term number we're looking for
  let aₙ : ℕ := a₁ + (n - 1) * d  -- formula for nth term of arithmetic sequence
  aₙ = 79 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l3557_355752
