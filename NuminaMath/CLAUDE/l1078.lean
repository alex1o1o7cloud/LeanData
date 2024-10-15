import Mathlib

namespace NUMINAMATH_CALUDE_extreme_values_of_f_l1078_107850

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- Theorem statement
theorem extreme_values_of_f :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 3 ∧ f b = -17 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l1078_107850


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1078_107874

/-- Given that the line y = x - 1 is tangent to the curve y = e^(x+a), prove that a = -2 --/
theorem tangent_line_to_exponential_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ - 1 = Real.exp (x₀ + a) ∧ 1 = Real.exp (x₀ + a)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1078_107874


namespace NUMINAMATH_CALUDE_sally_plums_l1078_107868

theorem sally_plums (melanie_plums dan_plums total_plums : ℕ) 
  (h1 : melanie_plums = 4)
  (h2 : dan_plums = 9)
  (h3 : total_plums = 16)
  (h4 : ∃ sally_plums : ℕ, melanie_plums + dan_plums + sally_plums = total_plums) :
  ∃ sally_plums : ℕ, sally_plums = 3 ∧ melanie_plums + dan_plums + sally_plums = total_plums := by
  sorry

end NUMINAMATH_CALUDE_sally_plums_l1078_107868


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1078_107833

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 3 / 4 = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1078_107833


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l1078_107816

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 998 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  70 * n % 350 = 210 % 350 ∧
  ∀ (m : ℕ), m < 1000 → m > 99 → 70 * m % 350 = 210 % 350 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l1078_107816


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l1078_107846

/-- Represents the digit reversal of a natural number -/
def digitReversal (n : ℕ) : ℕ := sorry

/-- Theorem stating that the difference between a natural number and its digit reversal is divisible by 9 -/
theorem difference_divisible_by_nine (n : ℕ) : 
  ∃ k : ℤ, (n : ℤ) - (digitReversal n : ℤ) = 9 * k := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l1078_107846


namespace NUMINAMATH_CALUDE_sum_of_triangles_l1078_107802

/-- The triangle operation that sums three numbers -/
def triangle (a b c : ℕ) : ℕ := a + b + c

/-- The first given triangle -/
def triangle1 : ℕ × ℕ × ℕ := (2, 3, 5)

/-- The second given triangle -/
def triangle2 : ℕ × ℕ × ℕ := (3, 4, 6)

/-- Theorem stating that the sum of triangle operations for both given triangles equals 23 -/
theorem sum_of_triangles :
  triangle triangle1.1 triangle1.2.1 triangle1.2.2 +
  triangle triangle2.1 triangle2.2.1 triangle2.2.2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_l1078_107802


namespace NUMINAMATH_CALUDE_only_C_is_lying_l1078_107891

-- Define the possible scores
def possible_scores : List ℕ := [1, 3, 5, 7, 9]

-- Define a structure for a person's statement
structure Statement where
  shots : ℕ
  hits : ℕ
  total_score : ℕ

-- Define the statements of A, B, C, and D
def statement_A : Statement := ⟨5, 5, 35⟩
def statement_B : Statement := ⟨6, 6, 36⟩
def statement_C : Statement := ⟨3, 3, 24⟩
def statement_D : Statement := ⟨4, 3, 21⟩

-- Define a function to check if a statement is valid
def is_valid_statement (s : Statement) (scores : List ℕ) : Prop :=
  ∃ (score_list : List ℕ),
    score_list.length = s.hits ∧
    score_list.sum = s.total_score ∧
    ∀ x ∈ score_list, x ∈ scores

-- Theorem stating that C's statement is false while others are true
theorem only_C_is_lying :
  is_valid_statement statement_A possible_scores ∧
  is_valid_statement statement_B possible_scores ∧
  ¬is_valid_statement statement_C possible_scores ∧
  is_valid_statement statement_D possible_scores :=
sorry

end NUMINAMATH_CALUDE_only_C_is_lying_l1078_107891


namespace NUMINAMATH_CALUDE_wheat_bags_theorem_l1078_107882

/-- Represents the deviation of each bag from the standard weight -/
def deviations : List Int := [-6, -3, -1, 7, 3, 4, -3, -2, -2, 1]

/-- The number of bags -/
def num_bags : Nat := 10

/-- The standard weight per bag in kg -/
def standard_weight : Int := 150

/-- The sum of all deviations -/
def total_deviation : Int := deviations.sum

/-- The average weight per bag -/
noncomputable def average_weight : ℚ := 
  (num_bags * standard_weight + total_deviation) / num_bags

theorem wheat_bags_theorem : 
  total_deviation = -2 ∧ average_weight = 149.8 := by sorry

end NUMINAMATH_CALUDE_wheat_bags_theorem_l1078_107882


namespace NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l1078_107835

theorem multiplicative_inverse_mod_million : ∃ N : ℕ, 
  (N > 0) ∧ 
  (N < 1000000) ∧ 
  ((123456 * 769230 * N) % 1000000 = 1) ∧ 
  (N = 1053) := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l1078_107835


namespace NUMINAMATH_CALUDE_midpoint_chain_l1078_107811

theorem midpoint_chain (A B C D E F G : ℝ) : 
  C = (A + B) / 2 →
  D = (A + C) / 2 →
  E = (A + D) / 2 →
  F = (A + E) / 2 →
  G = (A + F) / 2 →
  G - A = 5 →
  B - A = 160 := by
sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1078_107811


namespace NUMINAMATH_CALUDE_notebook_cost_l1078_107894

/-- The cost of a purchase given the number of notebooks, number of pencils, and total paid -/
def purchase_cost (notebooks : ℕ) (pencils : ℕ) (total_paid : ℚ) : ℚ := total_paid

/-- The theorem stating the cost of each notebook -/
theorem notebook_cost :
  ∀ (notebook_price pencil_price : ℚ),
    purchase_cost 5 4 20 - 3.5 = 5 * notebook_price + 4 * pencil_price →
    purchase_cost 2 2 7 = 2 * notebook_price + 2 * pencil_price →
    notebook_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1078_107894


namespace NUMINAMATH_CALUDE_number_difference_proof_l1078_107856

theorem number_difference_proof (x : ℝ) : x - (3 / 5) * x = 50 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l1078_107856


namespace NUMINAMATH_CALUDE_pike_eel_fat_difference_l1078_107803

theorem pike_eel_fat_difference (herring_fat eel_fat : ℕ) (pike_fat : ℕ) 
  (fish_count : ℕ) (total_fat : ℕ) : 
  herring_fat = 40 →
  eel_fat = 20 →
  pike_fat > eel_fat →
  fish_count = 40 →
  fish_count * herring_fat + fish_count * eel_fat + fish_count * pike_fat = total_fat →
  total_fat = 3600 →
  pike_fat - eel_fat = 10 := by
sorry

end NUMINAMATH_CALUDE_pike_eel_fat_difference_l1078_107803


namespace NUMINAMATH_CALUDE_unique_valid_arrangement_l1078_107839

/-- Represents the positions in the hexagon --/
inductive Position
| A | B | C | D | E | F

/-- Represents a line in the hexagon --/
structure Line where
  p1 : Position
  p2 : Position
  p3 : Position

/-- The arrangement of digits in the hexagon --/
def Arrangement := Position → Fin 6

/-- The 7 lines in the hexagon --/
def lines : List Line := [
  ⟨Position.A, Position.B, Position.C⟩,
  ⟨Position.A, Position.D, Position.F⟩,
  ⟨Position.A, Position.E, Position.F⟩,
  ⟨Position.B, Position.C, Position.D⟩,
  ⟨Position.B, Position.E, Position.D⟩,
  ⟨Position.C, Position.E, Position.F⟩,
  ⟨Position.D, Position.E, Position.F⟩
]

/-- Check if an arrangement is valid --/
def isValidArrangement (arr : Arrangement) : Prop :=
  (∀ p : Position, arr p ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ p q : Position, p ≠ q → arr p ≠ arr q) ∧
  (∀ l : Line, (arr l.p1).val + (arr l.p2).val + (arr l.p3).val = 15)

/-- The unique valid arrangement --/
def uniqueArrangement : Arrangement :=
  fun p => match p with
  | Position.A => 4
  | Position.B => 1
  | Position.C => 2
  | Position.D => 5
  | Position.E => 6
  | Position.F => 3

theorem unique_valid_arrangement :
  isValidArrangement uniqueArrangement ∧
  (∀ arr : Arrangement, isValidArrangement arr → arr = uniqueArrangement) := by
  sorry


end NUMINAMATH_CALUDE_unique_valid_arrangement_l1078_107839


namespace NUMINAMATH_CALUDE_probability_multiple_5_or_7_l1078_107809

def is_multiple_of_5_or_7 (n : ℕ) : Bool :=
  n % 5 = 0 || n % 7 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_5_or_7 |>.length

theorem probability_multiple_5_or_7 :
  count_multiples 50 / 50 = 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_5_or_7_l1078_107809


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l1078_107878

-- Define the number of each type of clothing
def num_hats : ℕ := 3
def num_shirts : ℕ := 4
def num_shorts : ℕ := 5
def num_socks : ℕ := 6

-- Define the total number of articles
def total_articles : ℕ := num_hats + num_shirts + num_shorts + num_socks

-- Define the number of articles to be drawn
def draw_count : ℕ := 4

-- Theorem statement
theorem probability_of_specific_draw :
  (num_hats.choose 1 * num_shirts.choose 1 * num_shorts.choose 1 * num_socks.choose 1) /
  (total_articles.choose draw_count) = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l1078_107878


namespace NUMINAMATH_CALUDE_minimum_jellybeans_l1078_107883

theorem minimum_jellybeans : ∃ n : ℕ,
  n ≥ 150 ∧
  n % 17 = 15 ∧
  (∀ m : ℕ, m ≥ 150 → m % 17 = 15 → n ≤ m) ∧
  n = 151 :=
by sorry

end NUMINAMATH_CALUDE_minimum_jellybeans_l1078_107883


namespace NUMINAMATH_CALUDE_product_of_sums_equals_power_specific_product_equals_power_l1078_107837

theorem product_of_sums_equals_power (a b : ℕ) :
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) * 
  (a^16 + b^16) * (a^32 + b^32) * (a^64 + b^64) = (a + b)^127 :=
by
  sorry

theorem specific_product_equals_power :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 9^127 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_power_specific_product_equals_power_l1078_107837


namespace NUMINAMATH_CALUDE_base_b_problem_l1078_107855

/-- Given that 1325 in base b is equal to the square of 35 in base b, prove that b = 10 in base 10 -/
theorem base_b_problem (b : ℕ) : 
  (3 * b + 5)^2 = b^3 + 3 * b^2 + 2 * b + 5 → b = 10 :=
by sorry

end NUMINAMATH_CALUDE_base_b_problem_l1078_107855


namespace NUMINAMATH_CALUDE_de_morgans_laws_l1078_107822

universe u

theorem de_morgans_laws {U : Type u} (A B : Set U) :
  (Set.compl (A ∪ B) = Set.compl A ∩ Set.compl B) ∧
  (Set.compl (A ∩ B) = Set.compl A ∪ Set.compl B) := by
  sorry

end NUMINAMATH_CALUDE_de_morgans_laws_l1078_107822


namespace NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l1078_107884

theorem quadratic_root_implies_v_value :
  ∀ v : ℝ,
  ((-15 - Real.sqrt 469) / 6 : ℝ) ∈ {x : ℝ | 3 * x^2 + 15 * x + v = 0} →
  v = -122/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l1078_107884


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1078_107849

def polynomial (x : ℝ) : ℝ := -2 * (x^7 - x^4 + 3*x^2 - 5) + 4*(x^3 + 2*x) - 3*(x^5 - 4)

theorem sum_of_coefficients : 
  (polynomial 1) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1078_107849


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l1078_107871

theorem snooker_ticket_difference :
  ∀ (vip_price general_price : ℚ) 
    (total_tickets : ℕ) 
    (total_cost : ℚ) 
    (vip_count general_count : ℕ),
  vip_price = 45 →
  general_price = 20 →
  total_tickets = 320 →
  total_cost = 7500 →
  vip_count + general_count = total_tickets →
  vip_price * vip_count + general_price * general_count = total_cost →
  general_count - vip_count = 232 := by
sorry

end NUMINAMATH_CALUDE_snooker_ticket_difference_l1078_107871


namespace NUMINAMATH_CALUDE_box_comparison_l1078_107815

-- Define a structure for a box with three dimensions
structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the relation "smaller than" for boxes
def smaller (k p : Box) : Prop :=
  (k.x ≤ p.x ∧ k.y ≤ p.y ∧ k.z ≤ p.z) ∨
  (k.x ≤ p.x ∧ k.y ≤ p.z ∧ k.z ≤ p.y) ∨
  (k.x ≤ p.y ∧ k.y ≤ p.x ∧ k.z ≤ p.z) ∨
  (k.x ≤ p.y ∧ k.y ≤ p.z ∧ k.z ≤ p.x) ∨
  (k.x ≤ p.z ∧ k.y ≤ p.x ∧ k.z ≤ p.y) ∨
  (k.x ≤ p.z ∧ k.y ≤ p.y ∧ k.z ≤ p.x)

-- Define boxes A, B, and C
def A : Box := ⟨5, 6, 3⟩
def B : Box := ⟨1, 5, 4⟩
def C : Box := ⟨2, 2, 3⟩

-- Theorem to prove A > B and C < A
theorem box_comparison : smaller B A ∧ smaller C A := by
  sorry


end NUMINAMATH_CALUDE_box_comparison_l1078_107815


namespace NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l1078_107819

-- Define the set of numbers
def numbers : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the concept of odd and even numbers
def isOdd (n : Nat) : Prop := n % 2 = 1
def isEven (n : Nat) : Prop := n % 2 = 0

-- Define the events
def event1 (a b : Nat) : Prop := (isOdd a ∧ isEven b) ∨ (isEven a ∧ isOdd b)
def event2 (a b : Nat) : Prop := isOdd a ∨ isOdd b
def event3 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (isEven a ∧ isEven b)
def event4 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (isEven a ∨ isEven b)

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : (Nat → Nat → Prop)) : Prop :=
  ∀ a b, a ∈ numbers → b ∈ numbers → ¬(e1 a b ∧ e2 a b)

-- Theorem statement
theorem only_event3_mutually_exclusive :
  (mutuallyExclusive event1 event3) ∧
  (¬mutuallyExclusive event1 event1) ∧
  (¬mutuallyExclusive event2 event4) ∧
  (¬mutuallyExclusive event4 event4) :=
sorry


end NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l1078_107819


namespace NUMINAMATH_CALUDE_at_least_one_hit_probability_l1078_107889

theorem at_least_one_hit_probability 
  (prob_A prob_B prob_C : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.5) 
  (h_C : prob_C = 0.4) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hit_probability_l1078_107889


namespace NUMINAMATH_CALUDE_person_B_age_l1078_107864

theorem person_B_age 
  (avg_ABC : (age_A + age_B + age_C) / 3 = 22)
  (avg_AB : (age_A + age_B) / 2 = 18)
  (avg_BC : (age_B + age_C) / 2 = 25)
  : age_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_person_B_age_l1078_107864


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l1078_107841

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l1078_107841


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1078_107897

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of a hyperbola -/
structure Foci where
  F₁ : Point
  F₂ : Point

/-- Checks if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Calculates the angle between three points -/
noncomputable def angle (p₁ p₂ p₃ : Point) : ℝ := sorry

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (f : Foci) (p : Point) :
  on_hyperbola h p →
  angle f.F₁ p f.F₂ = Real.pi / 2 →
  2 * angle p f.F₁ f.F₂ = angle p f.F₂ f.F₁ →
  eccentricity h = Real.sqrt 3 + 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1078_107897


namespace NUMINAMATH_CALUDE_lamp_cost_theorem_l1078_107862

-- Define the prices of lamps
def price_A : ℝ := sorry
def price_B : ℝ := sorry

-- Define the total number of lamps
def total_lamps : ℕ := 200

-- Define the function for total cost
def total_cost (a : ℕ) : ℝ := sorry

-- Theorem statement
theorem lamp_cost_theorem :
  -- Conditions
  (3 * price_A + 5 * price_B = 50) ∧
  (price_A + 3 * price_B = 26) ∧
  (∀ a : ℕ, total_cost a = price_A * a + price_B * (total_lamps - a)) →
  -- Conclusions
  (price_A = 5 ∧ price_B = 7) ∧
  (∀ a : ℕ, total_cost a = -2 * a + 1400) ∧
  (total_cost 80 = 1240) := by
  sorry


end NUMINAMATH_CALUDE_lamp_cost_theorem_l1078_107862


namespace NUMINAMATH_CALUDE_sum_equality_implies_k_value_l1078_107896

/-- Given a real number k > 1 satisfying the infinite sum equation, prove k equals the specified value. -/
theorem sum_equality_implies_k_value (k : ℝ) 
  (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 2) : 
  k = 2 + 1.5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_implies_k_value_l1078_107896


namespace NUMINAMATH_CALUDE_min_value_fraction_l1078_107875

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → 1/a + a/(8*b) ≤ 1/x + x/(8*y)) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ 1/x + x/(8*y) = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1078_107875


namespace NUMINAMATH_CALUDE_geometry_propositions_l1078_107836

-- Define the type for planes
variable (Plane : Type)

-- Define the type for lines
variable (Line : Type)

-- Define the relation for two planes being distinct
variable (distinct : Plane → Plane → Prop)

-- Define the relation for two lines intersecting
variable (intersect : Line → Line → Prop)

-- Define the relation for a line being within a plane
variable (within : Line → Plane → Prop)

-- Define the relation for two lines being parallel
variable (parallel_lines : Line → Line → Prop)

-- Define the relation for two planes being parallel
variable (parallel_planes : Plane → Plane → Prop)

-- Define the relation for a line being perpendicular to a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation for a line being perpendicular to another line
variable (perp_lines : Line → Line → Prop)

-- Define the relation for two planes intersecting along a line
variable (intersect_along : Plane → Plane → Line → Prop)

-- Define the relation for two planes being perpendicular
variable (perp_planes : Plane → Plane → Prop)

-- Define the relation for a line being outside a plane
variable (outside : Line → Plane → Prop)

theorem geometry_propositions 
  (α β : Plane) 
  (h_distinct : distinct α β) :
  (∀ (l1 l2 m1 m2 : Line), 
    intersect l1 l2 ∧ within l1 α ∧ within l2 α ∧ 
    within m1 β ∧ within m2 β ∧ 
    parallel_lines l1 m1 ∧ parallel_lines l2 m2 → 
    parallel_planes α β) ∧ 
  (∃ (l : Line) (m1 m2 : Line), 
    perp_line_plane l α ∧ 
    within m1 α ∧ within m2 α ∧ intersect m1 m2 ∧ 
    perp_lines l m1 ∧ perp_lines l m2 ∧ 
    ¬(∀ (n : Line), within n α ∧ perp_lines l n → perp_line_plane l α)) ∧
  (∃ (l m : Line), 
    intersect_along α β l ∧ within m α ∧ perp_lines m l ∧ ¬perp_planes α β) ∧
  (∀ (l m : Line), 
    outside l α ∧ within m α ∧ parallel_lines l m → 
    ∀ (n : Line), within n α → ¬intersect l n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1078_107836


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1078_107808

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 2*x - 3 > 0 ↔ x > 3 ∨ x < -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1078_107808


namespace NUMINAMATH_CALUDE_item_list_price_l1078_107807

/-- The list price of an item -/
def list_price : ℝ := 33

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Charles's selling price -/
def charles_price (x : ℝ) : ℝ := x - 18

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.15

/-- Charles's commission rate -/
def charles_rate : ℝ := 0.18

theorem item_list_price :
  alice_rate * alice_price list_price = charles_rate * charles_price list_price :=
by sorry

end NUMINAMATH_CALUDE_item_list_price_l1078_107807


namespace NUMINAMATH_CALUDE_percentage_study_both_math_and_sociology_l1078_107854

theorem percentage_study_both_math_and_sociology :
  ∀ (S : ℕ) (So Ma Bi MaSo : ℕ),
    S = 200 →
    So = (56 * S) / 100 →
    Ma = (44 * S) / 100 →
    Bi = (40 * S) / 100 →
    Bi - (S - So - Ma + MaSo) ≤ 60 →
    MaSo ≤ Bi - 60 →
    (MaSo * 100) / S = 10 :=
by sorry

end NUMINAMATH_CALUDE_percentage_study_both_math_and_sociology_l1078_107854


namespace NUMINAMATH_CALUDE_inequality_theorem_l1078_107885

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2)⁻¹ ≤ (x₁ * y₁ - z₁^2)⁻¹ + (x₂ * y₂ - z₂^2)⁻¹ ∧
  (((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2)⁻¹ = (x₁ * y₁ - z₁^2)⁻¹ + (x₂ * y₂ - z₂^2)⁻¹ ↔ 
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1078_107885


namespace NUMINAMATH_CALUDE_problem_solution_l1078_107830

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (h1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 5)
  (h2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 20)
  (h3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 145) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 380 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1078_107830


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l1078_107893

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 5}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1^2 + 6*p.1 - 8}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define the line y = x - 1
def center_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the lines l
def line_l1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 12*p.1 - 5*p.2 = 0}

theorem circle_and_line_theorem :
  -- The center of circle_C lies on center_line
  (∃ c : ℝ × ℝ, c ∈ center_line ∧ ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = 5) ∧
  -- circle_C passes through the intersection of parabola and x_axis
  (∀ p : ℝ × ℝ, p ∈ parabola ∩ x_axis → p ∈ circle_C) ∧
  -- For any line through origin intersecting circle_C at M and N with ON = 2OM,
  -- the line is either line_l1 or line_l2
  (∀ l : Set (ℝ × ℝ), origin ∈ l →
    (∃ M N : ℝ × ℝ, M ∈ l ∩ circle_C ∧ N ∈ l ∩ circle_C ∧ 
      N.1 = 2*M.1 ∧ N.2 = 2*M.2) →
    l = line_l1 ∨ l = line_l2) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l1078_107893


namespace NUMINAMATH_CALUDE_system_solution_l1078_107879

theorem system_solution (x y k : ℝ) : 
  x + 2*y = 2*k ∧ 
  2*x + y = 4*k ∧ 
  x + y = 4 → 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1078_107879


namespace NUMINAMATH_CALUDE_initial_liquid_a_amount_l1078_107812

/-- Given a mixture of liquids A and B with an initial ratio and a replacement process,
    calculate the initial amount of liquid A. -/
theorem initial_liquid_a_amount
  (initial_ratio_a : ℚ)
  (initial_ratio_b : ℚ)
  (replacement_amount : ℚ)
  (final_ratio_a : ℚ)
  (final_ratio_b : ℚ)
  (h_initial_ratio : initial_ratio_a / initial_ratio_b = 4 / 1)
  (h_replacement : replacement_amount = 20)
  (h_final_ratio : final_ratio_a / final_ratio_b = 2 / 3)
  : initial_ratio_a * (initial_ratio_a + initial_ratio_b) / (initial_ratio_a + initial_ratio_b) = 16 := by
  sorry


end NUMINAMATH_CALUDE_initial_liquid_a_amount_l1078_107812


namespace NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l1078_107831

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y ∧ y < z)
  (h2 : y - x > 9)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z) :
  ∀ w : ℤ, w ≥ 13 ∧ (∃ (a b c : ℤ), a < b ∧ b < c ∧ b - a > 9 ∧ Even a ∧ Odd b ∧ Odd c ∧ c - a = w) →
  z - x ≥ w :=
by sorry

end NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l1078_107831


namespace NUMINAMATH_CALUDE_possible_a_values_l1078_107881

theorem possible_a_values (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 5, x^2 - 6*x + 2 - a > 0) →
  (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_possible_a_values_l1078_107881


namespace NUMINAMATH_CALUDE_min_value_of_function_l1078_107820

theorem min_value_of_function :
  ∀ x : ℝ, x^2 + 1 / (x^2 + 1) + 3 ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1078_107820


namespace NUMINAMATH_CALUDE_sector_area_l1078_107867

/-- The area of a circular sector with central angle 120° and radius 4 is 16π/3 -/
theorem sector_area : 
  let central_angle : ℝ := 120
  let radius : ℝ := 4
  let sector_area : ℝ := (central_angle * π * radius^2) / 360
  sector_area = 16 * π / 3 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1078_107867


namespace NUMINAMATH_CALUDE_goods_train_passing_time_goods_train_passing_time_approx_9_seconds_l1078_107804

/-- The time taken for a goods train to pass a man in another train -/
theorem goods_train_passing_time (passenger_train_speed goods_train_speed : ℝ) 
  (goods_train_length : ℝ) : ℝ :=
  let relative_speed := passenger_train_speed + goods_train_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  goods_train_length / relative_speed_mps

/-- Proof that the time taken is approximately 9 seconds -/
theorem goods_train_passing_time_approx_9_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |goods_train_passing_time 60 52 280 - 9| < ε :=
sorry

end NUMINAMATH_CALUDE_goods_train_passing_time_goods_train_passing_time_approx_9_seconds_l1078_107804


namespace NUMINAMATH_CALUDE_ellipse_intersection_properties_l1078_107813

-- Define the line and ellipse
def line (x y : ℝ) : Prop := y = -x + 1
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the intersection points
def intersectionPoints (a b : ℝ) : Prop := ∃ A B : ℝ × ℝ, 
  line A.1 A.2 ∧ line B.1 B.2 ∧ ellipse A.1 A.2 a b ∧ ellipse B.1 B.2 a b

-- Define eccentricity and focal length
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 3 / 3
def focalLength (c : ℝ) : Prop := c = 1

-- Define perpendicularity of OA and OB
def perpendicular (A B : ℝ × ℝ) : Prop := A.1 * B.1 + A.2 * B.2 = 0

-- Main theorem
theorem ellipse_intersection_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : intersectionPoints a b) :
  (∃ A B : ℝ × ℝ, 
    eccentricity ((a^2 - b^2) / a^2) ∧ 
    focalLength ((a^2 - b^2) / 2) → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 3 / 5) ∧
  (∃ A B : ℝ × ℝ,
    perpendicular A B → 
    (1/2 : ℝ) ≤ ((a^2 - b^2) / a^2) ∧ ((a^2 - b^2) / a^2) ≤ Real.sqrt 2 / 2 →
    2 * a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_properties_l1078_107813


namespace NUMINAMATH_CALUDE_quadratic_polynomial_prime_values_l1078_107866

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial := ℤ → ℤ

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := sorry

/-- Predicate to check if a polynomial takes prime values at three consecutive integer points -/
def TakesPrimeValuesAtThreeConsecutivePoints (f : QuadraticPolynomial) : Prop :=
  ∃ n : ℤ, IsPrime (f (n - 1)) ∧ IsPrime (f n) ∧ IsPrime (f (n + 1))

/-- Predicate to check if a polynomial takes a prime value at least at one more integer point -/
def TakesPrimeValueAtOneMorePoint (f : QuadraticPolynomial) : Prop :=
  ∃ m : ℤ, (∀ n : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1) → IsPrime (f m)

/-- Theorem stating that if a quadratic polynomial with integer coefficients takes prime values
    at three consecutive integer points, then it takes a prime value at least at one more integer point -/
theorem quadratic_polynomial_prime_values (f : QuadraticPolynomial) :
  TakesPrimeValuesAtThreeConsecutivePoints f → TakesPrimeValueAtOneMorePoint f :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_prime_values_l1078_107866


namespace NUMINAMATH_CALUDE_odd_divisor_of_3n_plus_1_l1078_107873

theorem odd_divisor_of_3n_plus_1 (n : ℕ) :
  n ≥ 1 ∧ Odd n ∧ n ∣ (3^n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisor_of_3n_plus_1_l1078_107873


namespace NUMINAMATH_CALUDE_apples_in_refrigerator_l1078_107876

def initial_apples : ℕ := 62
def pie_apples : ℕ := initial_apples / 2
def muffin_apples : ℕ := 6

def refrigerator_apples : ℕ := initial_apples - pie_apples - muffin_apples

theorem apples_in_refrigerator : refrigerator_apples = 25 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_refrigerator_l1078_107876


namespace NUMINAMATH_CALUDE_lada_elevator_speed_ratio_l1078_107845

/-- The ratio of Lada's original speed to the elevator's speed -/
def speed_ratio : ℚ := 11/4

/-- The number of floors in the first scenario -/
def floors_first : ℕ := 3

/-- The number of floors in the second scenario -/
def floors_second : ℕ := 7

/-- The factor by which Lada increases her speed in the second scenario -/
def speed_increase : ℚ := 2

/-- The factor by which Lada's waiting time increases in the second scenario -/
def wait_time_increase : ℚ := 3

theorem lada_elevator_speed_ratio :
  ∀ (V U : ℚ) (S : ℝ),
  V > 0 → U > 0 → S > 0 →
  (floors_second : ℚ) / (speed_increase * U) - floors_second / V = 
    wait_time_increase * (floors_first / U - floors_first / V) →
  U / V = speed_ratio := by sorry

end NUMINAMATH_CALUDE_lada_elevator_speed_ratio_l1078_107845


namespace NUMINAMATH_CALUDE_min_value_theorem_l1078_107880

-- Define the quadratic inequality solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, (a * x^2 + 2 * x + b > 0) ↔ (x ≠ -1/a)

-- Define the theorem
theorem min_value_theorem (a b : ℝ) (h1 : solution_set a b) (h2 : a > b) :
  ∃ min_val : ℝ, min_val = 2 * Real.sqrt 2 ∧
  ∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1078_107880


namespace NUMINAMATH_CALUDE_three_heads_with_tail_probability_l1078_107826

/-- A fair coin flip sequence that ends when either three heads in a row or two tails in a row occur -/
inductive CoinFlipSequence
  | Incomplete : List Bool → CoinFlipSequence
  | ThreeHeads : List Bool → CoinFlipSequence
  | TwoTails : List Bool → CoinFlipSequence

/-- The probability of getting three heads in a row with at least one tail before the third head -/
def probability_three_heads_with_tail : ℚ :=
  5 / 64

/-- The main theorem stating that the calculated probability is correct -/
theorem three_heads_with_tail_probability :
  probability_three_heads_with_tail = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_with_tail_probability_l1078_107826


namespace NUMINAMATH_CALUDE_task_completion_time_l1078_107888

theorem task_completion_time 
  (time_A : ℝ) (time_B : ℝ) (time_C : ℝ) 
  (rest_A : ℝ) (rest_B : ℝ) :
  time_A = 10 →
  time_B = 15 →
  time_C = 15 →
  rest_A = 1 →
  rest_B = 2 →
  (1 - rest_B / time_B - (1 / time_A + 1 / time_B) * rest_A) / 
  (1 / time_A + 1 / time_B + 1 / time_C) + rest_A + rest_B = 29/7 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_l1078_107888


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1078_107865

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 3 →
    length / width = 3 →
    2 * r = width →
    length * width = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1078_107865


namespace NUMINAMATH_CALUDE_rain_duration_theorem_l1078_107818

def rain_duration_day1 : ℕ := 10

def rain_duration_day2 (d1 : ℕ) : ℕ := d1 + 2

def rain_duration_day3 (d2 : ℕ) : ℕ := 2 * d2

def total_rain_duration (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem rain_duration_theorem :
  total_rain_duration rain_duration_day1
    (rain_duration_day2 rain_duration_day1)
    (rain_duration_day3 (rain_duration_day2 rain_duration_day1)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_theorem_l1078_107818


namespace NUMINAMATH_CALUDE_ratio_simplification_l1078_107861

theorem ratio_simplification (a b c : ℝ) (n m p : ℕ+) 
  (h : a^(n : ℕ) / c^(p : ℕ) = 3 / 7 ∧ b^(m : ℕ) / c^(p : ℕ) = 4 / 7) :
  (a^(n : ℕ) + b^(m : ℕ) + c^(p : ℕ)) / c^(p : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_simplification_l1078_107861


namespace NUMINAMATH_CALUDE_parking_probability_l1078_107890

/-- Represents a parking lot -/
structure ParkingLot where
  totalSpaces : ℕ
  occupiedSpaces : ℕ

/-- Calculates the probability of finding a specified number of adjacent empty spaces -/
def probabilityOfAdjacentEmptySpaces (lot : ParkingLot) (requiredSpaces : ℕ) : ℚ :=
  sorry

theorem parking_probability (lot : ParkingLot) :
  lot.totalSpaces = 20 →
  lot.occupiedSpaces = 14 →
  probabilityOfAdjacentEmptySpaces lot 3 = 19/25 :=
by sorry

end NUMINAMATH_CALUDE_parking_probability_l1078_107890


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l1078_107825

/-- Given a triangle ABC where a, b, c are sides opposite to angles A, B, C respectively,
    if a² = b² + ¼c², then (a cos B) / c = 5/8 -/
theorem triangle_side_angle_relation (a b c : ℝ) (h : a^2 = b^2 + (1/4)*c^2) :
  (a * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) / c = 5/8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_angle_relation_l1078_107825


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1078_107851

theorem weight_of_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  n = 5 →
  avg_increase = 1.5 →
  replaced_weight = 65 →
  ∃ (new_weight : ℝ), new_weight = 72.5 ∧
    n * avg_increase = new_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1078_107851


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1078_107898

theorem trigonometric_inequality (x y z : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hz : 0 < z ∧ z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1078_107898


namespace NUMINAMATH_CALUDE_max_covered_squares_l1078_107887

def checkerboard_width : ℕ := 15
def checkerboard_height : ℕ := 36
def tile_side_1 : ℕ := 7
def tile_side_2 : ℕ := 5

theorem max_covered_squares :
  ∃ (n m : ℕ),
    n * (tile_side_1 ^ 2) + m * (tile_side_2 ^ 2) = checkerboard_width * checkerboard_height ∧
    ∀ (k l : ℕ),
      k * (tile_side_1 ^ 2) + l * (tile_side_2 ^ 2) ≤ checkerboard_width * checkerboard_height →
      k * (tile_side_1 ^ 2) + l * (tile_side_2 ^ 2) ≤ n * (tile_side_1 ^ 2) + m * (tile_side_2 ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_max_covered_squares_l1078_107887


namespace NUMINAMATH_CALUDE_power_equality_l1078_107877

theorem power_equality : (32 : ℕ)^4 * 4^5 = 2^30 := by sorry

end NUMINAMATH_CALUDE_power_equality_l1078_107877


namespace NUMINAMATH_CALUDE_train_length_l1078_107844

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 126 → time = 9 → speed * time * (1000 / 3600) = 315 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1078_107844


namespace NUMINAMATH_CALUDE_current_intensity_bound_l1078_107886

/-- Given a voltage and a minimum resistance, the current intensity is bounded above. -/
theorem current_intensity_bound (U R : ℝ) (hU : U = 200) (hR : R ≥ 62.5) :
  let I := U / R
  I ≤ 3.2 := by
  sorry

end NUMINAMATH_CALUDE_current_intensity_bound_l1078_107886


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_29_div_9_l1078_107847

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determinant of a 3x3 matrix -/
def det3 (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

/-- Two lines are coplanar if the determinant of their direction vectors and the vector between their points is zero -/
def areCoplanar (l1 l2 : Line3D) : Prop :=
  let (x1, y1, z1) := l1.point
  let (x2, y2, z2) := l2.point
  let v := (x2 - x1, y2 - y1, z2 - z1)
  det3 l1.direction l2.direction v = 0

/-- The main theorem -/
theorem lines_coplanar_iff_k_eq_neg_29_div_9 :
  let l1 : Line3D := ⟨(3, 2, 4), (2, -1, 3)⟩
  let l2 : Line3D := ⟨(0, 4, 1), (3*k, 1, 2)⟩
  areCoplanar l1 l2 ↔ k = -29/9 := by
  sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_29_div_9_l1078_107847


namespace NUMINAMATH_CALUDE_inequality_solution_l1078_107852

theorem inequality_solution (a b c d : ℝ) : 
  (∀ x : ℝ, ((x - a) * (x - b) * (x - c)) / (x - d) ≤ 0 ↔ 
    (x < -4 ∨ (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26))) →
  a < b →
  b < c →
  a + 3*b + 3*c + 4*d = 72 := by
sorry


end NUMINAMATH_CALUDE_inequality_solution_l1078_107852


namespace NUMINAMATH_CALUDE_money_loses_exchange_value_valid_money_properties_l1078_107821

/-- Represents an individual on an island -/
structure Individual where
  name : String

/-- Represents money found on the island -/
structure Money where
  amount : ℕ

/-- Represents the state of being on a deserted island -/
structure DesertedIsland where
  inhabitants : List Individual

/-- Function to determine if money has value as a medium of exchange -/
def hasExchangeValue (island : DesertedIsland) (money : Money) : Prop :=
  island.inhabitants.length > 1

/-- Theorem stating that money loses its exchange value on a deserted island with only one inhabitant -/
theorem money_loses_exchange_value 
  (crusoe : Individual) 
  (island : DesertedIsland) 
  (money : Money) 
  (h1 : island.inhabitants = [crusoe]) : 
  ¬(hasExchangeValue island money) := by
  sorry

/-- Properties required for an item to be considered money -/
structure MoneyProperties where
  durability : Prop
  portability : Prop
  divisibility : Prop
  acceptability : Prop
  uniformity : Prop
  limitedSupply : Prop

/-- Function to determine if an item can be considered money -/
def isValidMoney (item : MoneyProperties) : Prop :=
  item.durability ∧ 
  item.portability ∧ 
  item.divisibility ∧ 
  item.acceptability ∧ 
  item.uniformity ∧ 
  item.limitedSupply

/-- Theorem stating that an item must possess all required properties to be considered valid money -/
theorem valid_money_properties (item : MoneyProperties) :
  isValidMoney item ↔ 
    (item.durability ∧ 
     item.portability ∧ 
     item.divisibility ∧ 
     item.acceptability ∧ 
     item.uniformity ∧ 
     item.limitedSupply) := by
  sorry

end NUMINAMATH_CALUDE_money_loses_exchange_value_valid_money_properties_l1078_107821


namespace NUMINAMATH_CALUDE_prob_diana_wins_is_half_l1078_107800

/-- Diana's die has 8 sides -/
def diana_sides : ℕ := 8

/-- Apollo's die has 6 sides -/
def apollo_sides : ℕ := 6

/-- The set of possible outcomes for Diana -/
def diana_outcomes : Finset ℕ := Finset.range diana_sides

/-- The set of possible outcomes for Apollo -/
def apollo_outcomes : Finset ℕ := Finset.range apollo_sides

/-- The set of even outcomes for Apollo -/
def apollo_even_outcomes : Finset ℕ := Finset.filter (fun n => n % 2 = 0) apollo_outcomes

/-- The probability that Diana rolls a number larger than Apollo, given that Apollo's number is even -/
def prob_diana_wins_given_apollo_even : ℚ :=
  let total_outcomes := (apollo_even_outcomes.card * diana_outcomes.card : ℚ)
  let favorable_outcomes := (apollo_even_outcomes.sum fun a =>
    (diana_outcomes.filter (fun d => d > a)).card : ℚ)
  favorable_outcomes / total_outcomes

/-- The main theorem: The probability is 1/2 -/
theorem prob_diana_wins_is_half : prob_diana_wins_given_apollo_even = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_diana_wins_is_half_l1078_107800


namespace NUMINAMATH_CALUDE_dress_savings_theorem_l1078_107870

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_money_needed := dress_cost - initial_savings
  let weekly_savings := weekly_allowance - weekly_spending
  (additional_money_needed + weekly_savings - 1) / weekly_savings

theorem dress_savings_theorem (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ)
  (h1 : dress_cost = 80)
  (h2 : initial_savings = 20)
  (h3 : weekly_allowance = 30)
  (h4 : weekly_spending = 10) :
  weeks_to_save dress_cost initial_savings weekly_allowance weekly_spending = 3 := by
  sorry

end NUMINAMATH_CALUDE_dress_savings_theorem_l1078_107870


namespace NUMINAMATH_CALUDE_mothers_age_l1078_107828

theorem mothers_age (certain_age : ℕ) (mothers_age : ℕ) : 
  mothers_age = 3 * certain_age → 
  certain_age + mothers_age = 40 → 
  mothers_age = 30 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l1078_107828


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l1078_107863

theorem complex_sum_equals_negative_two (w : ℂ) : 
  w = Complex.cos (3 * Real.pi / 8) + Complex.I * Complex.sin (3 * Real.pi / 8) →
  2 * (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l1078_107863


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1078_107823

/-- A quadratic function f(x) = ax^2 + 2bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_order : c > b ∧ b > a

/-- The graph of f passes through (1, 0) -/
def passes_through_one_zero (f : QuadraticFunction) : Prop :=
  f.a + 2 * f.b + f.c = 0

/-- The graph of f intersects with y = -a -/
def intersects_neg_a (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, f.a * x^2 + 2 * f.b * x + f.c = -f.a

/-- The ratio b/a is in [0, 1) -/
def ratio_in_range (f : QuadraticFunction) : Prop :=
  0 ≤ f.b / f.a ∧ f.b / f.a < 1

/-- Line segments AB, BC, CD form an obtuse triangle -/
def forms_obtuse_triangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB + CD > BC ∧ AB^2 + CD^2 < BC^2

/-- The ratio b/a is in the specified range -/
def ratio_in_specific_range (f : QuadraticFunction) : Prop :=
  -1 + 4/21 < f.b / f.a ∧ f.b / f.a < -1 + Real.sqrt 15 / 3

theorem quadratic_function_properties (f : QuadraticFunction)
    (h_pass : passes_through_one_zero f)
    (h_intersect : intersects_neg_a f) :
  ratio_in_range f ∧
  (∀ A B C D : ℝ × ℝ, forms_obtuse_triangle A B C D →
    ratio_in_specific_range f) :=
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1078_107823


namespace NUMINAMATH_CALUDE_calculate_markup_l1078_107824

/-- Calculate the markup for an article given its purchase price, overhead percentage, and required net profit. -/
theorem calculate_markup (purchase_price overhead_percentage net_profit : ℝ) : 
  purchase_price = 48 → 
  overhead_percentage = 0.20 → 
  net_profit = 12 → 
  let overhead_cost := purchase_price * overhead_percentage
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  let markup := selling_price - purchase_price
  markup = 21.60 := by
sorry


end NUMINAMATH_CALUDE_calculate_markup_l1078_107824


namespace NUMINAMATH_CALUDE_count_scalene_triangles_l1078_107869

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ 
  a + b + c < 15 ∧
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_scalene_triangles : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    S.card = 6 ∧ 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2) :=
sorry

end NUMINAMATH_CALUDE_count_scalene_triangles_l1078_107869


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1078_107840

theorem complex_number_theorem (a : ℝ) (z : ℂ) (h1 : z = (a^2 - 1) + (a + 1) * I) 
  (h2 : z.re = 0) : (a + I^2016) / (1 + I) = 1 - I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1078_107840


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1078_107817

def M : Set ℝ := {x | (2 - x) / (x + 1) ≥ 0}
def N : Set ℝ := {x | ∃ y, y = Real.log x}

theorem set_intersection_equality : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1078_107817


namespace NUMINAMATH_CALUDE_smallest_n_for_sock_arrangement_l1078_107838

theorem smallest_n_for_sock_arrangement : 
  (∃ n : ℕ, n > 0 ∧ (n + 1) * (n + 2) / 2 > 1000000 ∧ 
   ∀ m : ℕ, m > 0 → (m + 1) * (m + 2) / 2 > 1000000 → m ≥ n) →
  (∃ n : ℕ, n > 0 ∧ (n + 1) * (n + 2) / 2 > 1000000 ∧ 
   ∀ m : ℕ, m > 0 → (m + 1) * (m + 2) / 2 > 1000000 → m ≥ n ∧ n = 1413) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sock_arrangement_l1078_107838


namespace NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1078_107834

/-- The sum of the interior angle of a regular pentagon and the interior angle of a regular triangle is 168°. -/
theorem pentagon_triangle_angle_sum : 
  let pentagon_angle : ℝ := 180 * (5 - 2) / 5
  let triangle_angle : ℝ := 180 * (3 - 2) / 3
  pentagon_angle + triangle_angle = 168 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1078_107834


namespace NUMINAMATH_CALUDE_prob_two_green_apples_l1078_107829

/-- The probability of selecting two green apples from a set of 8 apples,
    where 4 are green, when choosing 2 apples at random. -/
theorem prob_two_green_apples (total : ℕ) (green : ℕ) (choose : ℕ) 
    (h_total : total = 8) 
    (h_green : green = 4) 
    (h_choose : choose = 2) : 
    Nat.choose green choose / Nat.choose total choose = 3 / 14 := by
  sorry

#check prob_two_green_apples

end NUMINAMATH_CALUDE_prob_two_green_apples_l1078_107829


namespace NUMINAMATH_CALUDE_family_age_problem_l1078_107872

theorem family_age_problem (family_size : ℕ) (current_average_age : ℝ) (youngest_age : ℝ) :
  family_size = 5 →
  current_average_age = 20 →
  youngest_age = 10 →
  let total_age : ℝ := family_size * current_average_age
  let other_members_age : ℝ := total_age - youngest_age
  let age_reduction : ℝ := (family_size - 1) * youngest_age
  let total_age_at_birth : ℝ := other_members_age - age_reduction
  let average_age_at_birth : ℝ := total_age_at_birth / family_size
  average_age_at_birth = 10 := by
sorry

end NUMINAMATH_CALUDE_family_age_problem_l1078_107872


namespace NUMINAMATH_CALUDE_tadpoles_kept_calculation_l1078_107810

/-- The number of tadpoles Trent kept, given the initial number and percentage released -/
def tadpoles_kept (x : ℝ) : ℝ :=
  x * (1 - 0.825)

/-- Theorem stating that the number of tadpoles kept is 0.175 * x -/
theorem tadpoles_kept_calculation (x : ℝ) :
  tadpoles_kept x = 0.175 * x := by
  sorry

end NUMINAMATH_CALUDE_tadpoles_kept_calculation_l1078_107810


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l1078_107801

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) : 
  cube_side = 10 → 
  ball_radius = 3 → 
  ⌊(cube_side ^ 3) / ((4 / 3) * Real.pi * ball_radius ^ 3)⌋ = 8 := by
sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l1078_107801


namespace NUMINAMATH_CALUDE_populations_equal_after_16_years_l1078_107892

def village_x_initial_population : ℕ := 74000
def village_x_decrease_rate : ℕ := 1200
def village_y_initial_population : ℕ := 42000
def village_y_increase_rate : ℕ := 800

def population_equal_time : ℕ := 16

theorem populations_equal_after_16_years :
  village_x_initial_population - population_equal_time * village_x_decrease_rate =
  village_y_initial_population + population_equal_time * village_y_increase_rate :=
by sorry

end NUMINAMATH_CALUDE_populations_equal_after_16_years_l1078_107892


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1078_107895

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 3 → x^2 - 2*x > 0) ∧ 
  (∃ x, x^2 - 2*x > 0 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1078_107895


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l1078_107848

theorem three_digit_cube_divisible_by_16 :
  ∃! n : ℕ, 100 ≤ 64 * n^3 ∧ 64 * n^3 ≤ 999 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l1078_107848


namespace NUMINAMATH_CALUDE_correct_base_notation_l1078_107899

def is_valid_base_representation (digits : List Nat) (base : Nat) : Prop :=
  digits.all (· < base) ∧ digits.head! > 0

theorem correct_base_notation :
  is_valid_base_representation [7, 5, 1] 9 ∧
  ¬is_valid_base_representation [7, 5, 1] 7 ∧
  ¬is_valid_base_representation [0, 9, 5] 12 ∧
  ¬is_valid_base_representation [9, 0, 1] 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_base_notation_l1078_107899


namespace NUMINAMATH_CALUDE_system_solution_l1078_107806

theorem system_solution (x y : ℝ) :
  (4 * (Real.cos x)^2 - 4 * Real.cos x * (Real.cos (6 * x))^2 + (Real.cos (6 * x))^2 = 0) ∧
  (Real.sin x = Real.cos y) ↔
  (∃ (k n : ℤ),
    ((x = π / 3 + 2 * π * ↑k ∧ (y = π / 6 + 2 * π * ↑n ∨ y = -π / 6 + 2 * π * ↑n)) ∨
     (x = -π / 3 + 2 * π * ↑k ∧ (y = 5 * π / 6 + 2 * π * ↑n ∨ y = -5 * π / 6 + 2 * π * ↑n)))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1078_107806


namespace NUMINAMATH_CALUDE_total_donuts_three_days_l1078_107842

def monday_donuts : ℕ := 14

def tuesday_donuts : ℕ := monday_donuts / 2

def wednesday_donuts : ℕ := 4 * monday_donuts

theorem total_donuts_three_days : 
  monday_donuts + tuesday_donuts + wednesday_donuts = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_donuts_three_days_l1078_107842


namespace NUMINAMATH_CALUDE_not_all_diagonal_cells_good_l1078_107859

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 13
  col : Fin 13

/-- Represents the table -/
def Table := Fin 13 → Fin 13 → Fin 25

/-- Checks if a cell is "good" -/
def is_good (t : Table) (c : Cell) : Prop :=
  ∀ n : Fin 25, (∃! i : Fin 13, t i c.col = n) ∧ (∃! j : Fin 13, t c.row j = n)

/-- Represents the main diagonal -/
def main_diagonal : List Cell :=
  List.map (λ i => ⟨i, i⟩) (List.range 13)

/-- The theorem to be proved -/
theorem not_all_diagonal_cells_good (t : Table) : 
  ¬(∀ c ∈ main_diagonal, is_good t c) := by
  sorry


end NUMINAMATH_CALUDE_not_all_diagonal_cells_good_l1078_107859


namespace NUMINAMATH_CALUDE_two_over_a_necessary_not_sufficient_l1078_107814

theorem two_over_a_necessary_not_sufficient (a : ℝ) (h : a ≠ 0) :
  (∀ a, a ^ 2 > 4 → 2 / a < 1) ∧
  (∃ a, 2 / a < 1 ∧ a ^ 2 ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_two_over_a_necessary_not_sufficient_l1078_107814


namespace NUMINAMATH_CALUDE_sasha_remainder_l1078_107843

theorem sasha_remainder (n : ℕ) (a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  0 ≤ b ∧ b < 102 ∧
  0 ≤ d ∧ d < 103 ∧
  a + d = 20 →
  b = 20 :=
by sorry

end NUMINAMATH_CALUDE_sasha_remainder_l1078_107843


namespace NUMINAMATH_CALUDE_cassidy_grounding_period_l1078_107853

/-- Calculate the total grounding period for Cassidy --/
theorem cassidy_grounding_period :
  let initial_grounding : ℕ := 14
  let below_b_penalty : ℕ := 3
  let main_below_b : ℕ := 4
  let extra_below_b : ℕ := 2
  let a_grades : ℕ := 2
  let main_penalty := (main_below_b * below_b_penalty ^ 2 : ℚ)
  let extra_penalty := (extra_below_b * (below_b_penalty / 2) ^ 2 : ℚ)
  let additional_days := main_penalty + extra_penalty
  let reduced_initial := initial_grounding - a_grades
  let total_days := reduced_initial + additional_days
  ⌈total_days⌉ = 53 := by sorry

end NUMINAMATH_CALUDE_cassidy_grounding_period_l1078_107853


namespace NUMINAMATH_CALUDE_specific_field_planted_fraction_l1078_107858

/-- Represents a right-angled triangular field with an unplanted square at the right angle. -/
structure TriangularField where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- Calculates the fraction of the field that is planted. -/
def planted_fraction (field : TriangularField) : ℚ :=
  sorry

/-- Theorem stating that for a specific field configuration, the planted fraction is 7/10. -/
theorem specific_field_planted_fraction :
  let field : TriangularField := { leg1 := 5, leg2 := 12, square_distance := 3 }
  planted_fraction field = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_specific_field_planted_fraction_l1078_107858


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l1078_107860

theorem complex_in_first_quadrant (z : ℂ) : z = Complex.mk (Real.sqrt 3) 1 → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l1078_107860


namespace NUMINAMATH_CALUDE_two_percent_as_decimal_l1078_107857

/-- Expresses a percentage as a decimal fraction -/
def percent_to_decimal (p : ℚ) : ℚ := p / 100

/-- Proves that 2% expressed as a decimal fraction is equal to 0.02 -/
theorem two_percent_as_decimal : percent_to_decimal 2 = 0.02 := by sorry

end NUMINAMATH_CALUDE_two_percent_as_decimal_l1078_107857


namespace NUMINAMATH_CALUDE_division_problem_l1078_107832

theorem division_problem (number : ℕ) : 
  (number / 20 = 6) ∧ (number % 20 = 2) → number = 122 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1078_107832


namespace NUMINAMATH_CALUDE_sequence_theorem_l1078_107805

def sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (r p : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → r * (n - p) * S (n + 1) = n^2 * a n + (n^2 - n - 2) * a 1

theorem sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ) (r p : ℝ) 
  (h1 : |a 1| ≠ |a 2|)
  (h2 : r ≠ 0)
  (h3 : sequence_property a S r p) :
  (p = 1) ∧ 
  (¬ ∃ k : ℝ, k ≠ 1 ∧ k ≠ -1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = k * a n) ∧
  (r = 2 → ∃ d : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) = a n + d) :=
by sorry

end NUMINAMATH_CALUDE_sequence_theorem_l1078_107805


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l1078_107827

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_N_complement_M :
  N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l1078_107827
