import Mathlib

namespace NUMINAMATH_CALUDE_correct_quotient_proof_l816_81685

theorem correct_quotient_proof (dividend : ℕ) (wrong_divisor correct_divisor wrong_quotient : ℕ) 
  (h1 : dividend = wrong_divisor * wrong_quotient)
  (h2 : wrong_divisor = 121)
  (h3 : correct_divisor = 215)
  (h4 : wrong_quotient = 432) :
  dividend / correct_divisor = 243 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l816_81685


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l816_81623

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 160 * π / 180 → (n - 2) * π = n * θ) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l816_81623


namespace NUMINAMATH_CALUDE_plywood_cut_squares_l816_81662

/-- Represents the number of squares obtained from cutting a square plywood --/
def num_squares (side : ℕ) (cut_size1 cut_size2 : ℕ) (total_cut_length : ℕ) : ℕ :=
  sorry

/-- The theorem statement --/
theorem plywood_cut_squares :
  num_squares 50 10 20 280 = 16 :=
sorry

end NUMINAMATH_CALUDE_plywood_cut_squares_l816_81662


namespace NUMINAMATH_CALUDE_pizza_ratio_proof_l816_81653

theorem pizza_ratio_proof (total_slices : ℕ) (calories_per_slice : ℕ) (calories_eaten : ℕ) : 
  total_slices = 8 → 
  calories_per_slice = 300 → 
  calories_eaten = 1200 → 
  (calories_eaten / calories_per_slice : ℚ) / total_slices = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_ratio_proof_l816_81653


namespace NUMINAMATH_CALUDE_mandatoryQuestions_eq_13_l816_81657

/-- Represents a math competition with mandatory and optional questions -/
structure MathCompetition where
  totalQuestions : ℕ
  correctAnswers : ℕ
  totalScore : ℕ
  mandatoryCorrectPoints : ℕ
  mandatoryIncorrectPoints : ℕ
  optionalCorrectPoints : ℕ

/-- Calculates the number of mandatory questions in the competition -/
def mandatoryQuestions (comp : MathCompetition) : ℕ :=
  sorry

/-- Theorem stating that the number of mandatory questions is 13 -/
theorem mandatoryQuestions_eq_13 (comp : MathCompetition) 
  (h1 : comp.totalQuestions = 25)
  (h2 : comp.correctAnswers = 15)
  (h3 : comp.totalScore = 49)
  (h4 : comp.mandatoryCorrectPoints = 3)
  (h5 : comp.mandatoryIncorrectPoints = 2)
  (h6 : comp.optionalCorrectPoints = 5) :
  mandatoryQuestions comp = 13 := by
  sorry

end NUMINAMATH_CALUDE_mandatoryQuestions_eq_13_l816_81657


namespace NUMINAMATH_CALUDE_baker_pies_sold_l816_81614

theorem baker_pies_sold (cakes : ℕ) (cake_price pie_price total_earnings : ℚ) 
  (h1 : cakes = 453)
  (h2 : cake_price = 12)
  (h3 : pie_price = 7)
  (h4 : total_earnings = 6318) :
  (total_earnings - cakes * cake_price) / pie_price = 126 :=
by sorry

end NUMINAMATH_CALUDE_baker_pies_sold_l816_81614


namespace NUMINAMATH_CALUDE_conditional_probability_l816_81680

-- Define the probability measure z
variable (z : Set α → ℝ)

-- Define events x and y
variable (x y : Set α)

-- State the theorem
theorem conditional_probability
  (hx : z x = 0.02)
  (hy : z y = 0.10)
  (hxy : z (x ∩ y) = 0.10)
  (h_prob : ∀ A, 0 ≤ z A ∧ z A ≤ 1)
  (h_add : ∀ A B, z (A ∪ B) = z A + z B - z (A ∩ B))
  : z x / z y = 1 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_l816_81680


namespace NUMINAMATH_CALUDE_objects_meeting_probability_l816_81606

/-- The probability of two objects meeting on a coordinate plane -/
theorem objects_meeting_probability :
  let start_C : ℕ × ℕ := (0, 0)
  let start_D : ℕ × ℕ := (4, 6)
  let step_length : ℕ := 1
  let prob_C_right : ℚ := 1/2
  let prob_C_up : ℚ := 1/2
  let prob_D_left : ℚ := 1/2
  let prob_D_down : ℚ := 1/2
  ∃ (meeting_prob : ℚ), meeting_prob = 55/1024 :=
by sorry

end NUMINAMATH_CALUDE_objects_meeting_probability_l816_81606


namespace NUMINAMATH_CALUDE_inequality_relationship_l816_81617

theorem inequality_relationship (a b : ℝ) : 
  (∀ a b, a > b → a + 1 > b - 2) ∧ 
  (∃ a b, a + 1 > b - 2 ∧ ¬(a > b)) := by
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l816_81617


namespace NUMINAMATH_CALUDE_bedroom_difference_is_sixty_l816_81661

/-- The difference in square footage between two bedrooms -/
def bedroom_size_difference (total_size martha_size : ℝ) : ℝ :=
  (total_size - martha_size) - martha_size

/-- Theorem: Given the total size of two bedrooms and the size of one bedroom,
    prove that the difference between the two bedroom sizes is 60 sq ft -/
theorem bedroom_difference_is_sixty
  (total_size : ℝ)
  (martha_size : ℝ)
  (h1 : total_size = 300)
  (h2 : martha_size = 120) :
  bedroom_size_difference total_size martha_size = 60 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_difference_is_sixty_l816_81661


namespace NUMINAMATH_CALUDE_toy_store_order_l816_81636

theorem toy_store_order (stored_toys : ℕ) (storage_percentage : ℚ) (total_toys : ℕ) :
  stored_toys = 140 →
  storage_percentage = 7/10 →
  (storage_percentage * total_toys : ℚ) = stored_toys →
  total_toys = 200 := by
sorry

end NUMINAMATH_CALUDE_toy_store_order_l816_81636


namespace NUMINAMATH_CALUDE_factorization_equality_l816_81694

theorem factorization_equality (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l816_81694


namespace NUMINAMATH_CALUDE_landscape_breadth_l816_81603

/-- Given a rectangular landscape with a playground, proves that the breadth is 420 meters -/
theorem landscape_breadth (length breadth : ℝ) (playground_area : ℝ) : 
  breadth = 6 * length →
  playground_area = 4200 →
  playground_area = (1 / 7) * (length * breadth) →
  breadth = 420 := by
sorry

end NUMINAMATH_CALUDE_landscape_breadth_l816_81603


namespace NUMINAMATH_CALUDE_count_divisible_by_seven_l816_81659

theorem count_divisible_by_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.Icc 200 400)).card = 29 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_seven_l816_81659


namespace NUMINAMATH_CALUDE_expression_simplification_l816_81663

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^3 / ((a - b) * (a - c)) + (x + b)^3 / ((b - a) * (b - c)) + (x + c)^3 / ((c - a) * (c - b)) = a + b + c - 3*x :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l816_81663


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l816_81667

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 5 = 10) 
  (h2 : a 4 = 7) 
  (h3 : arithmetic_sequence a d) : 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l816_81667


namespace NUMINAMATH_CALUDE_villager_count_l816_81690

theorem villager_count (milk bottles_of_milk : ℕ) (apples : ℕ) (bread : ℕ) 
  (milk_left : ℕ) (apples_left : ℕ) (bread_short : ℕ) :
  bottles_of_milk = 160 →
  apples = 197 →
  bread = 229 →
  milk_left = 4 →
  apples_left = 2 →
  bread_short = 5 →
  ∃ (villagers : ℕ),
    villagers > 0 ∧
    (bottles_of_milk - milk_left) % villagers = 0 ∧
    (apples - apples_left) % villagers = 0 ∧
    (bread + bread_short) % villagers = 0 ∧
    villagers = 39 := by
  sorry

end NUMINAMATH_CALUDE_villager_count_l816_81690


namespace NUMINAMATH_CALUDE_fish_count_l816_81627

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 11

/-- The total number of fish Lilly and Rosy have -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 21 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l816_81627


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l816_81672

/-- The selling price of a cricket bat given profit and profit percentage -/
theorem cricket_bat_selling_price 
  (profit : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : profit = 205)
  (h2 : profit_percentage = 31.782945736434108) :
  let cost_price := profit / (profit_percentage / 100)
  let selling_price := cost_price + profit
  selling_price = 850 := by
sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l816_81672


namespace NUMINAMATH_CALUDE_cos_15_degrees_l816_81630

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_degrees_l816_81630


namespace NUMINAMATH_CALUDE_vector_bc_coordinates_l816_81648

/-- Given points A, B, and vector AC, prove that vector BC has specific coordinates -/
theorem vector_bc_coordinates (A B C : ℝ × ℝ) (h1 : A = (0, 1)) (h2 : B = (3, 2)) 
  (h3 : C.1 - A.1 = -4 ∧ C.2 - A.2 = -3) : 
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by
  sorry

#check vector_bc_coordinates

end NUMINAMATH_CALUDE_vector_bc_coordinates_l816_81648


namespace NUMINAMATH_CALUDE_negation_equivalence_l816_81618

theorem negation_equivalence :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, (n : ℝ) < x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l816_81618


namespace NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_420_l816_81632

theorem sum_distinct_prime_factors_of_420 : 
  (Finset.sum (Nat.factors 420).toFinset id) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_420_l816_81632


namespace NUMINAMATH_CALUDE_congruence_problem_l816_81610

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 20 = 3 → (3 * x + 14) % 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l816_81610


namespace NUMINAMATH_CALUDE_complex_multiplication_l816_81640

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l816_81640


namespace NUMINAMATH_CALUDE_sum_b_m_is_neg_eleven_fifths_l816_81674

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  m : ℚ
  b : ℚ
  x : ℚ
  y : ℚ
  h1 : y = m * x + 3
  h2 : y = 2 * x + b
  h3 : x = 5
  h4 : y = 7

/-- The sum of b and m for the intersecting lines -/
def sum_b_m (l : IntersectingLines) : ℚ := l.b + l.m

/-- Theorem stating that the sum of b and m is -11/5 -/
theorem sum_b_m_is_neg_eleven_fifths (l : IntersectingLines) : 
  sum_b_m l = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_m_is_neg_eleven_fifths_l816_81674


namespace NUMINAMATH_CALUDE_sin_arctan_equation_l816_81641

theorem sin_arctan_equation (y : ℝ) (hy : y > 0) 
  (h : Real.sin (Real.arctan y) = 1 / (2 * y)) : 
  y^2 = (1 + Real.sqrt 17) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_arctan_equation_l816_81641


namespace NUMINAMATH_CALUDE_sum_of_cubes_product_l816_81642

theorem sum_of_cubes_product : ∃ a b : ℤ, a^3 + b^3 = 35 ∧ a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_product_l816_81642


namespace NUMINAMATH_CALUDE_inverse_difference_inverse_l816_81639

theorem inverse_difference_inverse (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - z⁻¹)⁻¹ = x * z / (z - x) :=
by sorry

end NUMINAMATH_CALUDE_inverse_difference_inverse_l816_81639


namespace NUMINAMATH_CALUDE_arithmetic_sequence_of_powers_no_infinite_arithmetic_sequence_of_powers_l816_81624

/-- For any positive integer n, there exists an arithmetic sequence of n different elements 
    where every term is a power of a positive integer greater than 1. -/
theorem arithmetic_sequence_of_powers (n : ℕ+) : 
  ∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i, ∃ (b c : ℕ), c > 1 ∧ a i = b^c) ∧
    (∀ i, a (i + 1) = a i + d) :=
sorry

/-- There does not exist an infinite arithmetic sequence where every term is a power 
    of a positive integer greater than 1. -/
theorem no_infinite_arithmetic_sequence_of_powers : 
  ¬∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i, ∃ (b c : ℕ), c > 1 ∧ a i = b^c) ∧
    (∀ i, a (i + 1) = a i + d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_of_powers_no_infinite_arithmetic_sequence_of_powers_l816_81624


namespace NUMINAMATH_CALUDE_patricia_candies_l816_81665

theorem patricia_candies (initial final taken : ℕ) 
  (h1 : taken = 5)
  (h2 : final = 71)
  (h3 : initial = final + taken) : 
  initial = 76 := by
sorry

end NUMINAMATH_CALUDE_patricia_candies_l816_81665


namespace NUMINAMATH_CALUDE_min_distance_complex_l816_81608

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l816_81608


namespace NUMINAMATH_CALUDE_rain_probability_l816_81693

theorem rain_probability (p_day : ℝ) (p_consecutive : ℝ) 
  (h1 : p_day = 1/3)
  (h2 : p_consecutive = 1/5) :
  p_consecutive / p_day = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l816_81693


namespace NUMINAMATH_CALUDE_half_power_inequality_l816_81625

theorem half_power_inequality (a b : ℝ) (h : a > b) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l816_81625


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l816_81619

def f (x : ℝ) := x^3 - x + 3

theorem tangent_line_at_one : 
  ∃ (a b : ℝ), ∀ (x y : ℝ), 
    (y = f x ∧ x = 1) → (y = a * x + b) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l816_81619


namespace NUMINAMATH_CALUDE_factorization_equality_l816_81684

theorem factorization_equality (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l816_81684


namespace NUMINAMATH_CALUDE_marbles_selection_count_l816_81609

/-- The number of ways to choose 4 marbles from 8, with at least one red -/
def choose_marbles (total_marbles : ℕ) (red_marbles : ℕ) (choose : ℕ) : ℕ :=
  Nat.choose (total_marbles - red_marbles) (choose - 1)

/-- Theorem: There are 35 ways to choose 4 marbles from 8, with at least one red -/
theorem marbles_selection_count :
  choose_marbles 8 1 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_count_l816_81609


namespace NUMINAMATH_CALUDE_leopard_arrangement_l816_81699

theorem leopard_arrangement (n : ℕ) (h : n = 8) :
  (2 : ℕ) * Nat.factorial (n - 2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_leopard_arrangement_l816_81699


namespace NUMINAMATH_CALUDE_toms_rate_difference_l816_81658

/-- Proves that Tom's rate is 5 steps per minute faster than Matt's, given their relative progress --/
theorem toms_rate_difference (matt_rate : ℕ) (matt_steps : ℕ) (tom_steps : ℕ) :
  matt_rate = 20 →
  matt_steps = 220 →
  tom_steps = 275 →
  ∃ (tom_rate : ℕ), tom_rate = matt_rate + 5 ∧ tom_steps * matt_rate = matt_steps * tom_rate :=
by sorry

end NUMINAMATH_CALUDE_toms_rate_difference_l816_81658


namespace NUMINAMATH_CALUDE_vector_equation_l816_81675

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

theorem vector_equation : c = 3 • a - b := by sorry

end NUMINAMATH_CALUDE_vector_equation_l816_81675


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l816_81651

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →  -- arithmetic sequence
  (b 5 * b 6 = 21) →  -- given condition
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = 21) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l816_81651


namespace NUMINAMATH_CALUDE_cherry_difference_l816_81626

theorem cherry_difference (initial_cherries left_cherries : ℕ) 
  (h1 : initial_cherries = 16)
  (h2 : left_cherries = 6) :
  initial_cherries - left_cherries = 10 := by
  sorry

end NUMINAMATH_CALUDE_cherry_difference_l816_81626


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l816_81692

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 15) 
  (h2 : z + x = 18) 
  (h3 : x + y = 17) : 
  Real.sqrt (x * y * z * (x + y + z)) = 10 * Real.sqrt 70 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l816_81692


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l816_81622

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- An octagon formed by connecting midpoints of another octagon's sides -/
def MidpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of an octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The ratio of the area of the midpoint octagon to the area of the original regular octagon is 1/2 -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (MidpointOctagon o) / area o = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l816_81622


namespace NUMINAMATH_CALUDE_circle_line_theorem_l816_81637

/-- Two circles passing through a common point -/
structure TwoCircles where
  D1 : ℝ
  E1 : ℝ
  D2 : ℝ
  E2 : ℝ
  h1 : 2^2 + (-1)^2 + D1*2 + E1*(-1) - 3 = 0
  h2 : 2^2 + (-1)^2 + D2*2 + E2*(-1) - 3 = 0

/-- The equation of the line passing through (D1, E1) and (D2, E2) -/
def line_equation (c : TwoCircles) (x y : ℝ) : Prop :=
  2*x - y + 2 = 0

theorem circle_line_theorem (c : TwoCircles) :
  line_equation c c.D1 c.E1 ∧ line_equation c c.D2 c.E2 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_theorem_l816_81637


namespace NUMINAMATH_CALUDE_residue_of_negative_1035_mod_37_l816_81691

theorem residue_of_negative_1035_mod_37 :
  ∃ (r : ℤ), r ≥ 0 ∧ r < 37 ∧ -1035 ≡ r [ZMOD 37] ∧ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1035_mod_37_l816_81691


namespace NUMINAMATH_CALUDE_flute_players_count_l816_81683

/-- The number of people in an orchestra with specified instrument counts -/
structure Orchestra :=
  (total : ℕ)
  (drums : ℕ)
  (trombone : ℕ)
  (trumpet : ℕ)
  (french_horn : ℕ)
  (violinist : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (clarinet : ℕ)
  (conductor : ℕ)

/-- Theorem stating that the number of flute players is 4 -/
theorem flute_players_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.drums = 1)
  (h3 : o.trombone = 4)
  (h4 : o.trumpet = 2)
  (h5 : o.french_horn = 1)
  (h6 : o.violinist = 3)
  (h7 : o.cellist = 1)
  (h8 : o.contrabassist = 1)
  (h9 : o.clarinet = 3)
  (h10 : o.conductor = 1) :
  o.total - (o.drums + o.trombone + o.trumpet + o.french_horn + 
             o.violinist + o.cellist + o.contrabassist + 
             o.clarinet + o.conductor) = 4 := by
  sorry


end NUMINAMATH_CALUDE_flute_players_count_l816_81683


namespace NUMINAMATH_CALUDE_curve_classification_l816_81629

structure Curve where
  m : ℝ
  n : ℝ

def isEllipse (c : Curve) : Prop :=
  c.m > c.n ∧ c.n > 0

def isHyperbola (c : Curve) : Prop :=
  c.m * c.n < 0

def isTwoLines (c : Curve) : Prop :=
  c.m = 0 ∧ c.n > 0

theorem curve_classification (c : Curve) :
  (isEllipse c → ∃ foci : ℝ × ℝ, foci.1 = 0) ∧
  (isHyperbola c → ∃ k : ℝ, k^2 = -c.m / c.n) ∧
  (isTwoLines c → ∃ y₁ y₂ : ℝ, y₁ = -y₂ ∧ y₁^2 = 1 / c.n) :=
sorry

end NUMINAMATH_CALUDE_curve_classification_l816_81629


namespace NUMINAMATH_CALUDE_parallel_sum_diff_l816_81656

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Given vectors a and b in ℝ², if a + b is parallel to a - b, then the second component of b is 1 -/
theorem parallel_sum_diff (x : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (2, x)
  are_parallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_sum_diff_l816_81656


namespace NUMINAMATH_CALUDE_spade_calculation_l816_81604

/-- Define the ⊙ operation for real numbers -/
def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 7 ⊙ (2 ⊙ 3) = 24 -/
theorem spade_calculation : spade 7 (spade 2 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l816_81604


namespace NUMINAMATH_CALUDE_count_arrangements_11250_l816_81634

def digits : List Nat := [1, 1, 2, 5, 0]

def is_multiple_of_two (n : Nat) : Bool :=
  n % 2 = 0

def is_five_digit (n : Nat) : Bool :=
  n ≥ 10000 ∧ n < 100000

def count_valid_arrangements (ds : List Nat) : Nat :=
  sorry

theorem count_arrangements_11250 : 
  count_valid_arrangements digits = 24 := by sorry

end NUMINAMATH_CALUDE_count_arrangements_11250_l816_81634


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l816_81668

theorem fraction_to_decimal : (5 : ℚ) / 125 = (4 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l816_81668


namespace NUMINAMATH_CALUDE_age_relation_proof_l816_81621

/-- Represents the current ages and future time when Alex's age is thrice Ben's --/
structure AgeRelation where
  ben_age : ℕ
  alex_age : ℕ
  michael_age : ℕ
  future_years : ℕ

/-- The conditions of the problem --/
def age_conditions (ar : AgeRelation) : Prop :=
  ar.ben_age = 4 ∧
  ar.alex_age = ar.ben_age + 30 ∧
  ar.michael_age = ar.alex_age + 4 ∧
  ar.alex_age + ar.future_years = 3 * (ar.ben_age + ar.future_years)

/-- The theorem to prove --/
theorem age_relation_proof :
  ∃ (ar : AgeRelation), age_conditions ar ∧ ar.future_years = 11 :=
sorry

end NUMINAMATH_CALUDE_age_relation_proof_l816_81621


namespace NUMINAMATH_CALUDE_cat_walking_rate_l816_81670

/-- Given a cat's walking scenario with total time, resistance time, and distance walked,
    calculate the cat's walking rate in feet per minute. -/
theorem cat_walking_rate 
  (total_time : ℝ) 
  (resistance_time : ℝ) 
  (distance_walked : ℝ) 
  (h1 : total_time = 28) 
  (h2 : resistance_time = 20) 
  (h3 : distance_walked = 64) : 
  (distance_walked / (total_time - resistance_time)) = 8 := by
  sorry

#check cat_walking_rate

end NUMINAMATH_CALUDE_cat_walking_rate_l816_81670


namespace NUMINAMATH_CALUDE_not_perfect_square_l816_81645

theorem not_perfect_square : ∃ (n : ℕ), n = 6^2041 ∧
  (∀ (m : ℕ), m^2 ≠ n) ∧
  (∃ (a : ℕ), 3^2040 = a^2) ∧
  (∃ (b : ℕ), 7^2042 = b^2) ∧
  (∃ (c : ℕ), 8^2043 = c^2) ∧
  (∃ (d : ℕ), 9^2044 = d^2) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l816_81645


namespace NUMINAMATH_CALUDE_min_product_xyz_l816_81664

theorem min_product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq_one : x + y + z = 1) (z_eq_2x : z = 2 * x) (y_eq_3x : y = 3 * x) :
  x * y * z ≥ 1 / 36 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀ + y₀ + z₀ = 1 ∧ z₀ = 2 * x₀ ∧ y₀ = 3 * x₀ ∧ x₀ * y₀ * z₀ = 1 / 36 :=
by sorry

end NUMINAMATH_CALUDE_min_product_xyz_l816_81664


namespace NUMINAMATH_CALUDE_inequality_system_solution_l816_81628

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l816_81628


namespace NUMINAMATH_CALUDE_team_combinations_l816_81602

theorem team_combinations (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_l816_81602


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l816_81607

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region -/
inductive Region
  | A
  | B
  | C
  | D

/-- Represents the company's sales point distribution -/
structure SalesPointDistribution where
  total_points : Nat
  region_points : Region → Nat
  large_points_in_C : Nat

/-- Represents an investigation -/
structure Investigation where
  sample_size : Nat
  population_size : Nat

/-- Determines the appropriate sampling method for an investigation -/
def appropriate_sampling_method (dist : SalesPointDistribution) (inv : Investigation) : SamplingMethod :=
  sorry

/-- The company's actual sales point distribution -/
def company_distribution : SalesPointDistribution :=
  { total_points := 600,
    region_points := fun r => match r with
      | Region.A => 150
      | Region.B => 120
      | Region.C => 180
      | Region.D => 150,
    large_points_in_C := 20 }

/-- Investigation ① -/
def investigation_1 : Investigation :=
  { sample_size := 100,
    population_size := 600 }

/-- Investigation ② -/
def investigation_2 : Investigation :=
  { sample_size := 7,
    population_size := 20 }

/-- Theorem stating the appropriate sampling methods for the given investigations -/
theorem appropriate_sampling_methods :
  appropriate_sampling_method company_distribution investigation_1 = SamplingMethod.StratifiedSampling ∧
  appropriate_sampling_method company_distribution investigation_2 = SamplingMethod.SimpleRandomSampling :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l816_81607


namespace NUMINAMATH_CALUDE_abs_neg_two_fifths_l816_81654

theorem abs_neg_two_fifths : |(-2 : ℚ) / 5| = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_fifths_l816_81654


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l816_81682

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l816_81682


namespace NUMINAMATH_CALUDE_domino_puzzle_l816_81611

theorem domino_puzzle (visible_points : ℕ) (num_tiles : ℕ) (grid_size : ℕ) :
  visible_points = 37 →
  num_tiles = 8 →
  grid_size = 4 →
  ∃ (missing_points : ℕ),
    (visible_points + missing_points) % grid_size = 0 ∧
    missing_points ≤ 3 ∧
    ∀ (m : ℕ), m > missing_points →
      (visible_points + m) % grid_size ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_domino_puzzle_l816_81611


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l816_81612

/-- Proves that the interest rate at which A lent to B is 15% given the conditions --/
theorem interest_rate_calculation (principal : ℝ) (rate_B_to_C : ℝ) (time : ℝ) (B_gain : ℝ) 
  (h_principal : principal = 2000)
  (h_rate_B_to_C : rate_B_to_C = 17)
  (h_time : time = 4)
  (h_B_gain : B_gain = 160)
  : ∃ R : ℝ, R = 15 ∧ 
    principal * (rate_B_to_C / 100) * time - principal * (R / 100) * time = B_gain :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l816_81612


namespace NUMINAMATH_CALUDE_dodecagon_vertex_product_l816_81676

/-- Regular dodecagon in the complex plane -/
structure RegularDodecagon where
  center : ℂ
  vertex : ℂ

/-- The product of the complex representations of all vertices of a regular dodecagon -/
def vertexProduct (d : RegularDodecagon) : ℂ :=
  (d.center + 1)^12 - 1

/-- Theorem: The product of vertices of a regular dodecagon with center (2,1) and a vertex at (3,1) -/
theorem dodecagon_vertex_product :
  let d : RegularDodecagon := { center := 2 + 1*I, vertex := 3 + 1*I }
  vertexProduct d = -2926 - 3452*I :=
by
  sorry

end NUMINAMATH_CALUDE_dodecagon_vertex_product_l816_81676


namespace NUMINAMATH_CALUDE_remaining_length_is_90_cm_l816_81635

-- Define the initial length in meters
def initial_length : ℝ := 1

-- Define the erased length in centimeters
def erased_length : ℝ := 10

-- Theorem to prove
theorem remaining_length_is_90_cm :
  (initial_length * 100 - erased_length) = 90 := by
  sorry

end NUMINAMATH_CALUDE_remaining_length_is_90_cm_l816_81635


namespace NUMINAMATH_CALUDE_students_passing_both_tests_l816_81620

theorem students_passing_both_tests 
  (total : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both : ℕ) 
  (h1 : total = 50) 
  (h2 : passed_long_jump = 40) 
  (h3 : passed_shot_put = 31) 
  (h4 : failed_both = 4) :
  total - failed_both = passed_long_jump + passed_shot_put - 25 := by
sorry

end NUMINAMATH_CALUDE_students_passing_both_tests_l816_81620


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l816_81615

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4,5}

theorem complement_intersection_A_B : 
  (U \ (A ∩ B)) = {1,4,5,6,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l816_81615


namespace NUMINAMATH_CALUDE_b_months_is_five_l816_81616

/-- Represents the grazing arrangement for a pasture -/
structure GrazingArrangement where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℚ
  c_share : ℚ

/-- Calculates the number of months B's oxen grazed given a GrazingArrangement -/
def calculate_b_months (g : GrazingArrangement) : ℚ :=
  ((g.total_rent - g.c_share) * g.c_oxen * g.c_months - g.a_oxen * g.a_months * g.c_share) /
  (g.b_oxen * g.c_share)

/-- Theorem stating that B's oxen grazed for 5 months under the given conditions -/
theorem b_months_is_five (g : GrazingArrangement) 
    (h1 : g.a_oxen = 10) 
    (h2 : g.a_months = 7) 
    (h3 : g.b_oxen = 12) 
    (h4 : g.c_oxen = 15) 
    (h5 : g.c_months = 3) 
    (h6 : g.total_rent = 175) 
    (h7 : g.c_share = 45) : 
  calculate_b_months g = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_months_is_five_l816_81616


namespace NUMINAMATH_CALUDE_shaded_fraction_is_37_72_l816_81689

/-- Represents a digit drawn on the grid -/
inductive Digit
  | one
  | nine
  | eight

/-- Represents the grid with drawn digits -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)
  (digits : List Digit)

/-- Calculates the number of small squares occupied by a digit -/
def squaresOccupied (d : Digit) : Nat :=
  match d with
  | Digit.one => 8
  | Digit.nine => 15
  | Digit.eight => 16

/-- Calculates the total number of squares in the grid -/
def totalSquares (g : Grid) : Nat :=
  g.rows * g.cols

/-- Calculates the number of squares occupied by all digits -/
def occupiedSquares (g : Grid) : Nat :=
  g.digits.foldl (fun acc d => acc + squaresOccupied d) 0

/-- Represents the fraction of shaded area -/
def shadedFraction (g : Grid) : Rat :=
  occupiedSquares g / totalSquares g

theorem shaded_fraction_is_37_72 (g : Grid) 
  (h1 : g.rows = 18)
  (h2 : g.cols = 8)
  (h3 : g.digits = [Digit.one, Digit.nine, Digit.nine, Digit.eight]) :
  shadedFraction g = 37 / 72 := by
  sorry

#eval shadedFraction { rows := 18, cols := 8, digits := [Digit.one, Digit.nine, Digit.nine, Digit.eight] }

end NUMINAMATH_CALUDE_shaded_fraction_is_37_72_l816_81689


namespace NUMINAMATH_CALUDE_problem_solution_l816_81679

theorem problem_solution (a b : ℝ) 
  (sum_eq : a + b = 12) 
  (diff_sq_eq : a^2 - b^2 = 48) : 
  a - b = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l816_81679


namespace NUMINAMATH_CALUDE_triangle_properties_l816_81698

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a / (Real.sin A) = b / (Real.sin B) ∧ 
  b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition: b^2 + c^2 - a^2 = bc
  b^2 + c^2 - a^2 = b * c →
  -- Area of triangle is 3√3/2
  (1/2) * a * b * (Real.sin C) = (3 * Real.sqrt 3) / 2 →
  -- Given condition: sin C + √3 cos C = 2
  Real.sin C + Real.sqrt 3 * Real.cos C = 2 →
  -- Prove: A = π/3 and a = 3
  A = π/3 ∧ a = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l816_81698


namespace NUMINAMATH_CALUDE_fraction_independence_l816_81687

theorem fraction_independence (a b c a₁ b₁ c₁ : ℝ) (h₁ : a₁ ≠ 0) :
  (∀ x, (a * x^2 + b * x + c) / (a₁ * x^2 + b₁ * x + c₁) = (a / a₁)) ↔ 
  (a / a₁ = b / b₁ ∧ b / b₁ = c / c₁) :=
sorry

end NUMINAMATH_CALUDE_fraction_independence_l816_81687


namespace NUMINAMATH_CALUDE_remainder_7n_mod_3_l816_81696

theorem remainder_7n_mod_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_3_l816_81696


namespace NUMINAMATH_CALUDE_factory_production_rate_solve_factory_production_rate_l816_81631

/-- Proves that the daily production rate in the first year was 3650 televisions,
    given a 10% reduction in the second year and a total production of 3285 televisions
    in the second year. -/
theorem factory_production_rate : ℝ → Prop :=
  fun daily_rate =>
    let reduction_factor : ℝ := 0.9
    let second_year_production : ℝ := 3285
    daily_rate * reduction_factor * 365 = second_year_production →
    daily_rate = 3650

/-- The actual theorem statement -/
theorem solve_factory_production_rate : 
  ∃ (rate : ℝ), factory_production_rate rate :=
sorry

end NUMINAMATH_CALUDE_factory_production_rate_solve_factory_production_rate_l816_81631


namespace NUMINAMATH_CALUDE_gcd_8512_13832_l816_81686

theorem gcd_8512_13832 : Nat.gcd 8512 13832 = 1064 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8512_13832_l816_81686


namespace NUMINAMATH_CALUDE_hen_egg_production_l816_81681

/-- Given the following conditions:
    - There are 10 hens
    - Eggs are sold for $3 per dozen
    - In 4 weeks, $120 worth of eggs were sold
    Prove that each hen lays 12 eggs per week. -/
theorem hen_egg_production 
  (num_hens : ℕ) 
  (price_per_dozen : ℚ) 
  (weeks : ℕ) 
  (total_sales : ℚ) 
  (h1 : num_hens = 10)
  (h2 : price_per_dozen = 3)
  (h3 : weeks = 4)
  (h4 : total_sales = 120) :
  (total_sales / price_per_dozen * 12 / weeks / num_hens : ℚ) = 12 := by
sorry


end NUMINAMATH_CALUDE_hen_egg_production_l816_81681


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l816_81646

theorem sqrt_difference_equality : Real.sqrt (49 + 49) - Real.sqrt (36 + 25) = 7 * Real.sqrt 2 - Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l816_81646


namespace NUMINAMATH_CALUDE_min_value_quadratic_l816_81673

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 - 6*x + 10 ≥ 1) ∧ (∃ x, x^2 - 6*x + 10 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l816_81673


namespace NUMINAMATH_CALUDE_total_serving_time_l816_81678

def total_patients : ℕ := 12
def standard_serving_time : ℕ := 5
def special_needs_ratio : ℚ := 1/3
def special_needs_time_increase : ℚ := 1/5

theorem total_serving_time :
  let special_patients := total_patients * special_needs_ratio
  let standard_patients := total_patients - special_patients
  let special_serving_time := standard_serving_time * (1 + special_needs_time_increase)
  let total_time := standard_patients * standard_serving_time + special_patients * special_serving_time
  total_time = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_serving_time_l816_81678


namespace NUMINAMATH_CALUDE_students_on_bus_l816_81695

theorem students_on_bus (first_stop : ℕ) (second_stop : ℕ) 
  (h1 : first_stop = 39) (h2 : second_stop = 29) :
  first_stop + second_stop = 68 := by
  sorry

end NUMINAMATH_CALUDE_students_on_bus_l816_81695


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l816_81677

-- Define the universal set U
def U : Set ℝ := {x | x < 5}

-- Define the set A
def A : Set ℝ := {x | x - 2 ≤ 0}

-- State the theorem
theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {x | 2 < x ∧ x < 5} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l816_81677


namespace NUMINAMATH_CALUDE_complex_square_l816_81601

theorem complex_square : (1 + Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l816_81601


namespace NUMINAMATH_CALUDE_f_properties_l816_81671

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.log x - 2 / x + 1

-- Theorem statement
theorem f_properties :
  (∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f' x = 0) ∧
  (∀ x : ℝ, x > 0 → f x > 0) := by
  sorry

end

end NUMINAMATH_CALUDE_f_properties_l816_81671


namespace NUMINAMATH_CALUDE_abc_inequality_l816_81669

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l816_81669


namespace NUMINAMATH_CALUDE_farm_feet_count_l816_81638

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of animals (heads) in the farm -/
def Farm.totalAnimals (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def Farm.totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- Theorem: In a farm with 44 animals, if there are 24 hens, then the total number of feet is 128 -/
theorem farm_feet_count (f : Farm) (h1 : f.totalAnimals = 44) (h2 : f.hens = 24) : 
  f.totalFeet = 128 := by
  sorry

end NUMINAMATH_CALUDE_farm_feet_count_l816_81638


namespace NUMINAMATH_CALUDE_age_difference_l816_81605

theorem age_difference (X Y Z : ℕ) : X + Y = Y + Z + 12 → (X - Z : ℚ) / 10 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l816_81605


namespace NUMINAMATH_CALUDE_max_notebooks_is_eleven_l816_81643

/-- Represents the maximum number of notebooks that can be purchased with a given budget. -/
def max_notebooks (single_price : ℕ) (pack4_price : ℕ) (pack7_price : ℕ) (budget : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific pricing and budget, the maximum number of notebooks is 11. -/
theorem max_notebooks_is_eleven :
  max_notebooks 2 6 9 15 = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_notebooks_is_eleven_l816_81643


namespace NUMINAMATH_CALUDE_frog_jump_coordinates_l816_81652

def initial_point : ℝ × ℝ := (-1, 0)
def right_jump : ℝ := 2
def up_jump : ℝ := 2

def final_point (p : ℝ × ℝ) (right : ℝ) (up : ℝ) : ℝ × ℝ :=
  (p.1 + right, p.2 + up)

theorem frog_jump_coordinates :
  final_point initial_point right_jump up_jump = (1, 2) := by sorry

end NUMINAMATH_CALUDE_frog_jump_coordinates_l816_81652


namespace NUMINAMATH_CALUDE_solution_set_inequality_l816_81697

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo 50 60 : Set ℝ) = {x | (x - 50) * (60 - x) > 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l816_81697


namespace NUMINAMATH_CALUDE_x_zero_value_l816_81655

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 1) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l816_81655


namespace NUMINAMATH_CALUDE_horner_eval_23_l816_81647

def horner_polynomial (a b c d x : ℤ) : ℤ := ((a * x + b) * x + c) * x + d

theorem horner_eval_23 :
  let f : ℤ → ℤ := λ x => 7 * x^3 + 3 * x^2 - 5 * x + 11
  let horner : ℤ → ℤ := horner_polynomial 7 3 (-5) 11
  (∀ step : ℤ, step ≠ 85169 → (step = 7 ∨ step = 164 ∨ step = 3762 ∨ step = 86537)) ∧
  f 23 = horner 23 ∧
  f 23 = 86537 := by
sorry

end NUMINAMATH_CALUDE_horner_eval_23_l816_81647


namespace NUMINAMATH_CALUDE_remainder_theorem_l816_81649

theorem remainder_theorem : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l816_81649


namespace NUMINAMATH_CALUDE_ratio_range_l816_81613

-- Define the condition for the point (x,y)
def satisfies_condition (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 ≤ 0

-- Define the range for y/x
def in_range (r : ℝ) : Prop :=
  0 ≤ r ∧ r ≤ 4/3

-- Theorem statement
theorem ratio_range (x y : ℝ) (h : satisfies_condition x y) (hx : x ≠ 0) :
  in_range (y / x) :=
sorry

end NUMINAMATH_CALUDE_ratio_range_l816_81613


namespace NUMINAMATH_CALUDE_man_upstream_speed_l816_81660

/-- Given a man's speed in still water and his speed downstream, 
    calculates his speed upstream -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that for a man with speed 60 kmph in still water 
    and 65 kmph downstream, his upstream speed is 55 kmph -/
theorem man_upstream_speed :
  speed_upstream 60 65 = 55 := by
  sorry


end NUMINAMATH_CALUDE_man_upstream_speed_l816_81660


namespace NUMINAMATH_CALUDE_fuel_consumption_problem_l816_81644

/-- The fuel consumption problem for an aviation engineer --/
theorem fuel_consumption_problem 
  (fuel_per_person : ℝ) 
  (fuel_per_bag : ℝ) 
  (num_passengers : ℕ) 
  (num_crew : ℕ) 
  (bags_per_person : ℕ) 
  (total_fuel : ℝ) 
  (trip_distance : ℝ) 
  (h1 : fuel_per_person = 3)
  (h2 : fuel_per_bag = 2)
  (h3 : num_passengers = 30)
  (h4 : num_crew = 5)
  (h5 : bags_per_person = 2)
  (h6 : total_fuel = 106000)
  (h7 : trip_distance = 400) :
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let additional_fuel_per_mile := total_people * fuel_per_person + total_bags * fuel_per_bag
  let total_fuel_per_mile := total_fuel / trip_distance
  total_fuel_per_mile - additional_fuel_per_mile = 20 := by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_problem_l816_81644


namespace NUMINAMATH_CALUDE_machine_time_difference_l816_81633

/- Define the variables -/
variable (W : ℝ) -- Number of widgets
variable (X : ℝ) -- Rate of machine X in widgets per day
variable (Y : ℝ) -- Rate of machine Y in widgets per day

/- Define the conditions -/
axiom machine_X_rate : X = W / 6
axiom combined_rate : X + Y = 5 * W / 12
axiom machine_X_alone : 30 * X = 5 * W

/- State the theorem -/
theorem machine_time_difference : 
  W / X - W / Y = 2 := by sorry

end NUMINAMATH_CALUDE_machine_time_difference_l816_81633


namespace NUMINAMATH_CALUDE_complex_equation_solution_l816_81666

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_solution (m n : ℝ) (h : (m : ℂ) / (1 + i) = 1 - n * i) :
  (m : ℂ) + n * i = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l816_81666


namespace NUMINAMATH_CALUDE_segment_length_l816_81688

/-- Given a line segment AB with points P and Q on it, prove that AB has length 120 -/
theorem segment_length (A B P Q : Real) : 
  (∃ x y u v : Real,
    -- P divides AB in ratio 3:5
    5 * x = 3 * y ∧ 
    -- Q divides AB in ratio 2:3
    3 * u = 2 * v ∧ 
    -- P is closer to A than Q
    u = x + 3 ∧ 
    v = y - 3 ∧ 
    -- AB is the sum of its parts
    A + B = x + y) → 
  A + B = 120 := by
sorry

end NUMINAMATH_CALUDE_segment_length_l816_81688


namespace NUMINAMATH_CALUDE_parallel_line_equation_l816_81650

/-- Given a line L1 with equation 2x + y - 5 = 0 and a point P (1, -3),
    prove that the line L2 passing through P and parallel to L1
    has the equation 2x + y + 1 = 0 -/
theorem parallel_line_equation (x y : ℝ) :
  let L1 : ℝ × ℝ → Prop := λ (x, y) ↦ 2 * x + y - 5 = 0
  let P : ℝ × ℝ := (1, -3)
  let L2 : ℝ × ℝ → Prop := λ (x, y) ↦ 2 * x + y + 1 = 0
  (∀ (x₁ y₁ x₂ y₂ : ℝ), L1 (x₁, y₁) ∧ L1 (x₂, y₂) → (y₂ - y₁) = -2 * (x₂ - x₁)) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), L2 (x₁, y₁) ∧ L2 (x₂, y₂) → (y₂ - y₁) = -2 * (x₂ - x₁)) →
  L2 P →
  ∀ (a b : ℝ), (∀ (x y : ℝ), a * x + b * y + 1 = 0 ↔ L2 (x, y)) →
  a = 2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l816_81650


namespace NUMINAMATH_CALUDE_a_10_equals_1000_l816_81600

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let first_term := 2 * n - 1
    let last_term := first_term + 2 * (n - 1)
    n * (first_term + last_term) / 2

theorem a_10_equals_1000 : sequence_a 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_1000_l816_81600
