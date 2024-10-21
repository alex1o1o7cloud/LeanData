import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_projection_sum_special_triangle_l338_33802

/-- Triangle with sides a, b, c -/
structure Triangle (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- Centroid of a triangle -/
def Centroid (α : Type*) [LinearOrderedField α] (t : Triangle α) : α × α := (0, 0)

/-- Projection of a point onto a line segment -/
def Projection (α : Type*) [LinearOrderedField α] (p : α × α) (seg : α × α × α × α) : α × α := p

/-- Distance between two points -/
def dist (α : Type*) [LinearOrderedField α] (p1 p2 : α × α) : α := 0

/-- Sum of distances from centroid to its projections on sides -/
noncomputable def CentroidProjectionSum (α : Type*) [LinearOrderedField α] (t : Triangle α) : α :=
  let cent := Centroid α t
  let proj1 := Projection α cent (0, 0, t.b, t.c)
  let proj2 := Projection α cent (0, 0, t.a, t.c)
  let proj3 := Projection α cent (0, 0, t.a, t.b)
  dist α cent proj1 + dist α cent proj2 + dist α cent proj3

theorem centroid_projection_sum_special_triangle :
  let t : Triangle ℝ := { a := 4, b := 6, c := 5 }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ abs (CentroidProjectionSum ℝ t - 4.082) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_projection_sum_special_triangle_l338_33802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_B_l338_33817

-- Define the set of divisors of 60
def divisors_of_60 : Finset Nat := Finset.filter (fun d => 60 % d = 0) (Finset.range 61)

-- Define B as the product of the divisors of 60
noncomputable def B : Nat := (divisors_of_60.toList).prod

-- Theorem statement
theorem distinct_prime_factors_of_B : 
  (Nat.factors B).toFinset.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_B_l338_33817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_diverges_l338_33897

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ+, (1 : ℝ) / (k + 2 : ℝ) ^ n.val

/-- The statement that the sum of g(n) from n = 1 to infinity diverges -/
theorem sum_g_diverges : ¬ ∃ S : ℝ, HasSum g S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_diverges_l338_33897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_survival_duration_l338_33885

/-- Given the following conditions:
1. A group of 4 vampires, where the leader drains 5 people/week and others drain 5 people/week each
2. A pack of 5 werewolves, where the alpha eats 7 people/week and others eat 5 people/week each
3. A ghost that feeds on 2 people/week
4. A coven of 4 witches that sacrifices 3 people/week
5. A horde of 20 zombies that each eat 1 person/week
6. A village of 500 people

Prove that the village will last for 6 weeks. -/
theorem village_survival_duration (
  vampire_leader_consumption : ℕ := 5)
  (vampire_follower_consumption : ℕ := 5)
  (vampire_followers : ℕ := 3)
  (werewolf_alpha_consumption : ℕ := 7)
  (werewolf_follower_consumption : ℕ := 5)
  (werewolf_followers : ℕ := 4)
  (ghost_consumption : ℕ := 2)
  (witch_coven_size : ℕ := 4)
  (witch_sacrifice : ℕ := 3)
  (zombie_count : ℕ := 20)
  (zombie_consumption : ℕ := 1)
  (village_population : ℕ := 500) :
  village_population / (
    vampire_leader_consumption + vampire_follower_consumption * vampire_followers +
    werewolf_alpha_consumption + werewolf_follower_consumption * werewolf_followers +
    ghost_consumption +
    witch_sacrifice +
    zombie_count * zombie_consumption
  ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_survival_duration_l338_33885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l338_33826

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(2*x - 3) * (8 : ℝ)^(x + 1) = (1024 : ℝ)^2 :=
by
  -- Define 8 and 1024 in terms of powers of 2
  have h1 : (8 : ℝ) = (2 : ℝ)^3 := by norm_num
  have h2 : (1024 : ℝ) = (2 : ℝ)^10 := by norm_num
  
  -- Prove that x = 4 is the unique solution
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l338_33826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increase_intervals_and_a_range_l338_33853

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + Real.log x - a*x

def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem function_increase_intervals_and_a_range :
  (∃ I₁ I₂ : Set ℝ, I₁ = Set.Ioo 0 (1/2) ∧ I₂ = Set.Ioi 1 ∧
    (∀ x y, x ∈ I₁ ∪ I₂ → y ∈ I₁ ∪ I₂ → x < y → f 3 x < f 3 y)) ∧
  (is_increasing (f 3) (Set.Ioo 0 1) → 3 ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increase_intervals_and_a_range_l338_33853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_l338_33882

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the gain per year in a borrowing and lending transaction -/
noncomputable def gainPerYear (principal : ℝ) (borrowRate : ℝ) (lendRate : ℝ) (time : ℝ) : ℝ :=
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time

theorem transaction_gain (principal : ℝ) (borrowRate : ℝ) (lendRate : ℝ) (time : ℝ) 
    (h_principal : principal = 5000)
    (h_borrowRate : borrowRate = 0.04)
    (h_lendRate : lendRate = 0.05)
    (h_time : time = 2) :
  gainPerYear principal borrowRate lendRate time = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_l338_33882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_midline_intersection_length_l338_33866

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the perimeter of a triangle
noncomputable def Perimeter (A B C : ℝ × ℝ) : ℝ := sorry

-- Define an excircle of a triangle
def Excircle (ω : Set (ℝ × ℝ)) (A B C P Q : ℝ × ℝ) : Prop := True

-- Define the midpoint of a line segment
def Midpoint (M : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := True

-- Define the circumcircle of a triangle
def Circumcircle (c : Set (ℝ × ℝ)) (A P Q : ℝ × ℝ) : Prop := True

-- Define a line passing through two points
def Line (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop := True

-- Define the intersection of a line and a circle
def Intersect (X Y : ℝ × ℝ) (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := True

-- Define the length of a line segment
noncomputable def Length (A B : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem excircle_midline_intersection_length 
  (A B C P Q M N X Y : ℝ × ℝ) 
  (ω c l : Set (ℝ × ℝ)) :
  Triangle A B C →
  Perimeter A B C = 1 →
  Excircle ω A B C P Q →
  Midpoint M A B →
  Midpoint N A C →
  Line l M N →
  Circumcircle c A P Q →
  Intersect X Y l c →
  Length X Y = 1/2 := by
  sorry

#check excircle_midline_intersection_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_midline_intersection_length_l338_33866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_network_count_l338_33879

/-- Represents a railway network connecting cities -/
structure RailwayNetwork where
  cities : Fin 5
  roads : Fin 4

/-- Represents the configuration of a railway network -/
inductive NetworkConfig
  | CentralHub
  | ThreeConnections
  | LinearConnection

/-- The number of different railway networks possible -/
def num_railway_networks : ℕ := 125

/-- Function to check if three points are collinear -/
def collinear (a b c : Fin 5) : Prop := sorry

/-- The theorem stating the number of different railway networks -/
theorem railway_network_count :
  (∀ (n : RailwayNetwork), ∃ (c : NetworkConfig), true) →
  (∀ (a b c : Fin 5), a ≠ b → b ≠ c → a ≠ c → ¬ collinear a b c) →
  num_railway_networks = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_network_count_l338_33879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_widgets_produced_correct_answer_a_is_correct_answer_b_is_incorrect_answer_c_is_incorrect_answer_d_is_incorrect_answer_e_is_incorrect_l338_33806

/-- The number of widgets produced by a given number of machines in a given number of days. -/
def widgetsProduced (x y z w v : ℚ) : ℚ := (w * y * v) / (x * z)

/-- Theorem stating that the formula for widgets produced is correct. -/
theorem widgets_produced_correct (x y z w v : ℚ) (hx : x ≠ 0) (hz : z ≠ 0) :
  widgetsProduced x y z w v = (w * y * v) / (x * z) := by
  -- Unfold the definition of widgetsProduced
  unfold widgetsProduced
  -- The result follows directly from the definition
  rfl

/-- Proof that the answer (A) is correct. -/
theorem answer_a_is_correct (x y z w v : ℚ) (hx : x ≠ 0) (hz : z ≠ 0) :
  widgetsProduced x y z w v = (w * y * v) / (x * z) := by
  -- This is exactly what the widgets_produced_correct theorem states
  exact widgets_produced_correct x y z w v hx hz

-- The following shows that the other answer choices are not correct
theorem answer_b_is_incorrect (x y z w v : ℚ) (hx : x ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0) (hw : w ≠ 0) (hv : v ≠ 0) :
  widgetsProduced x y z w v ≠ (x * z) / (w * y * v) := by
  sorry -- Proof omitted for brevity

theorem answer_c_is_incorrect (x y z w v : ℚ) (hx : x ≠ 0) (hz : z ≠ 0) (hv : v ≠ 0) :
  widgetsProduced x y z w v ≠ (w * y * z) / v := by
  sorry -- Proof omitted for brevity

theorem answer_d_is_incorrect (x y z w v : ℚ) (hx : x ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  widgetsProduced x y z w v ≠ (y * x * v) / (w * z) := by
  sorry -- Proof omitted for brevity

theorem answer_e_is_incorrect (x y z w v : ℚ) (hx : x ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (hv : v ≠ 0) :
  widgetsProduced x y z w v ≠ (x * y * z) / (w * v) := by
  sorry -- Proof omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_widgets_produced_correct_answer_a_is_correct_answer_b_is_incorrect_answer_c_is_incorrect_answer_d_is_incorrect_answer_e_is_incorrect_l338_33806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lambda_for_triangle_inequality_l338_33861

theorem smallest_lambda_for_triangle_inequality (a b c : ℝ) 
    (triangle_cond : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
    (side_cond : a ≥ (b + c) / 3) :
  let optimal_lambda : ℝ := (2 * Real.sqrt 2 + 1) / 7
  ∀ μ : ℝ, μ > 0 → 
    (a * c + b * c - c^2 ≤ μ * (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c)) →
    optimal_lambda ≤ μ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lambda_for_triangle_inequality_l338_33861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l338_33819

/-- Represents a polynomial with integer coefficients -/
def MyPolynomial := List Int

/-- Horner's method for polynomial evaluation -/
def horner_eval (p : MyPolynomial) (x : Int) : Int × Nat × Nat :=
  p.foldl
    (fun (acc, mults, adds) coeff =>
      (acc * x + coeff, mults + 1, adds + 1))
    (0, 0, 0)

/-- Our specific polynomial 2x^7 + x^6 - 3x^3 + 2x + 1 -/
def f : MyPolynomial := [1, 2, 0, -3, 0, 0, 1, 2]

theorem horner_method_operations :
  let (result, mults, adds) := horner_eval f 2
  mults = 7 ∧ adds = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l338_33819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_bottom_width_l338_33871

/-- Represents the cross-section of a water channel -/
structure WaterChannel where
  topWidth : ℝ
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ

/-- Calculates the area of a trapezoidal cross-section -/
noncomputable def trapezoidArea (channel : WaterChannel) : ℝ :=
  (1/2) * (channel.topWidth + channel.bottomWidth) * channel.depth

/-- Theorem stating that given the specific measurements, the bottom width is 6 meters -/
theorem water_channel_bottom_width :
  ∃ (channel : WaterChannel),
    channel.topWidth = 12 ∧
    channel.depth = 70 ∧
    channel.area = 630 ∧
    trapezoidArea channel = channel.area ∧
    channel.bottomWidth = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_bottom_width_l338_33871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_mixture_concentration_l338_33809

/-- Represents a salt solution with a given volume and concentration -/
structure SaltSolution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of salt in a solution -/
noncomputable def saltAmount (solution : SaltSolution) : ℝ :=
  solution.volume * solution.concentration

/-- Represents the mixing of two salt solutions -/
noncomputable def mixSolutions (s1 s2 : SaltSolution) : SaltSolution :=
  { volume := s1.volume + s2.volume,
    concentration := (saltAmount s1 + saltAmount s2) / (s1.volume + s2.volume) }

theorem salt_mixture_concentration 
  (initial : SaltSolution) 
  (added : SaltSolution) :
  initial.volume = 30 ∧ 
  initial.concentration = 0.2 ∧
  added.volume = 30 ∧ 
  added.concentration = 0.6 →
  (mixSolutions initial added).concentration = 0.4 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_mixture_concentration_l338_33809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_three_l338_33849

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 9) / (x - 3)

-- Theorem stating that f has a vertical asymptote at x = 3
theorem vertical_asymptote_at_three :
  ∃ (L : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 →
    ∀ (x : ℝ), 0 < |x - 3| ∧ |x - 3| < δ → |f x| > L := by
  sorry

#check vertical_asymptote_at_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_three_l338_33849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_binary_palindromes_1988_l338_33808

/-- A natural number is a binary palindrome if its binary representation reads the same forwards and backwards. -/
def IsBinaryPalindrome (n : ℕ) : Prop :=
  let binary := n.digits 2
  binary = binary.reverse

/-- A decidable instance for IsBinaryPalindrome -/
instance (n : ℕ) : Decidable (IsBinaryPalindrome n) :=
  show Decidable (n.digits 2 = (n.digits 2).reverse) from inferInstance

/-- The count of binary palindromes not exceeding a given natural number. -/
def CountBinaryPalindromes (max : ℕ) : ℕ :=
  (Finset.range (max + 1)).filter IsBinaryPalindrome |>.card

/-- The theorem stating that the count of binary palindromes not exceeding 1988 is 92. -/
theorem count_binary_palindromes_1988 :
  CountBinaryPalindromes 1988 = 92 := by
  sorry

#eval CountBinaryPalindromes 1988

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_binary_palindromes_1988_l338_33808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l338_33839

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | Real.rpow 2 (x + 1) > 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l338_33839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l338_33856

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (y - f x) = 1 - x - y) →
  (∀ x : ℝ, f x = 1/2 - x) :=
by
  intro f h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l338_33856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_proof_l338_33868

theorem tan_value_proof (α : Real) 
  (h1 : α ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h2 : Real.sin α + Real.cos α = -1/5) : 
  Real.tan α = -4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_proof_l338_33868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_of_vectors_l338_33846

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem angle_cosine_of_vectors (u v : E) (h1 : ‖u‖ = 5) (h2 : ‖v‖ = 7) (h3 : ‖u + v‖ = 9) :
  inner u v / (‖u‖ * ‖v‖) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_of_vectors_l338_33846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_sequence_element_product_mod_5_l338_33833

/-- The sequence of numbers from 3 to 93, incrementing by 10 -/
def mySequence : List Nat := List.range 10 |>.map (fun i => 3 + 10 * i)

/-- The product of all numbers in the sequence -/
def myProduct : Nat := mySequence.prod

theorem remainder_of_sequence_element (n : Nat) : n ∈ mySequence → n % 5 = 3 := by sorry

theorem product_mod_5 : myProduct % 5 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_sequence_element_product_mod_5_l338_33833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pants_purchase_cost_l338_33843

theorem pants_purchase_cost 
  (number_of_pants : ℕ) 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (tax_rate : ℝ) : 
  (number_of_pants = 10 ∧ 
   original_price = 45 ∧ 
   discount_rate = 0.2 ∧ 
   tax_rate = 0.1) → 
  (let discounted_price := original_price * (1 - discount_rate)
   let total_before_tax := number_of_pants * discounted_price
   let total_after_tax := total_before_tax * (1 + tax_rate)
   total_after_tax = 396) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pants_purchase_cost_l338_33843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l338_33834

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x + y = 4

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- State the theorem
theorem min_distance_between_curves :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 ∧
    x₁ = 3/2 ∧ y₁ = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l338_33834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_perimeter_l338_33835

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Represents a folded rectangle -/
structure FoldedRectangle where
  rect : Rectangle
  U : Point
  V : Point
  Q' : Point

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def perimeter (r : Rectangle) : ℝ :=
  2 * (distance r.P r.Q + distance r.Q r.R)

theorem folded_rectangle_perimeter 
  (fr : FoldedRectangle)
  (h1 : distance fr.rect.P fr.U = 6)
  (h2 : distance fr.U fr.rect.Q = 20)
  (h3 : distance fr.rect.S fr.V = 4)
  (h4 : fr.Q'.y = fr.rect.S.y)
  (h5 : fr.U.x = fr.rect.P.x + 6)
  (h6 : fr.V.x = fr.rect.R.x - 6)
  (h7 : fr.U.y = fr.rect.P.y)
  (h8 : fr.V.y = fr.rect.R.y) :
  perimeter fr.rect = 320/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_perimeter_l338_33835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotone_l338_33874

/-- A power function f(x) = (m^2 - 9m + 19)x^(m-4) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 9*m + 19) * (x^(m-4))

/-- f is monotonically increasing on (0, +∞) -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f m x < f m y

theorem power_function_monotone (m : ℝ) :
  is_monotone_increasing m → m = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotone_l338_33874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_power_magnitude_of_specific_complex_magnitude_of_specific_complex_power_l338_33896

open Real Complex

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) : 
  Complex.abs (z^n) = (Complex.abs z)^n := by sorry

theorem magnitude_of_specific_complex : 
  Complex.abs (4/7 + 3/7 * I) = 5/7 := by sorry

theorem magnitude_of_specific_complex_power : 
  Complex.abs ((4/7 + 3/7 * I)^7) = (5/7)^7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_power_magnitude_of_specific_complex_magnitude_of_specific_complex_power_l338_33896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_3y_mod_9_l338_33863

theorem remainder_3y_mod_9 (y : ℕ) (h : y % 9 = 5) : (3 * y) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_3y_mod_9_l338_33863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_swaps_correct_l338_33878

/-- The minimum number of swaps needed to guarantee the ball falls into the hole -/
def min_swaps (n : ℕ) : ℕ :=
  2 * n^2 - n

/-- A function representing the minimum number of swaps needed to guarantee
    the ball falls into the hole for any initial configuration -/
def minimum_swaps_to_guarantee_ball_in_hole (n : ℕ) : ℕ :=
  2 * n^2 - n

/-- The theorem stating the minimum number of swaps needed -/
theorem min_swaps_correct (n : ℕ) (h : n > 0) :
  min_swaps n = minimum_swaps_to_guarantee_ball_in_hole n :=
by
  -- Unfold the definitions
  unfold min_swaps
  unfold minimum_swaps_to_guarantee_ball_in_hole
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_swaps_correct_l338_33878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l338_33832

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + (n * (n - 1) / 2) * seq.d

theorem arithmetic_sequence_sum_eight
  (seq : ArithmeticSequence)
  (h3 : 2 * seq.a 8 = 6 + seq.a 11)
  (h4 : ∃ r : ℚ, seq.a 4 = seq.a 3 * r ∧ seq.a 6 = seq.a 4 * r) :
  sum_n seq 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l338_33832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l338_33893

/-- Given a hyperbola with an asymptote y = √2x and sharing a common focus with the ellipse y²/4 + x² = 1,
    prove that its equation is y²/2 - x² = 1 -/
theorem hyperbola_equation 
  (h : Set (ℝ × ℝ)) 
  (e : Set (ℝ × ℝ)) 
  (asymptote : ℝ → ℝ)
  (is_asymptote : (x : ℝ) → (asymptote x, x) ∈ frontier h)
  (shared_focus : ∃ (f : ℝ × ℝ), f ∈ h ∧ f ∈ e)
  (ellipse_eq : ∀ x y : ℝ, (x, y) ∈ e ↔ y^2 / 4 + x^2 = 1) :
  (∀ x y : ℝ, (x, y) ∈ h ↔ y^2 / 2 - x^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l338_33893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_chemistry_count_l338_33812

theorem basketball_team_chemistry_count 
  (total_players : Finset ℕ) 
  (biology_set : Finset ℕ)
  (chemistry_set : Finset ℕ)
  (h1 : total_players.card = 20)
  (h2 : biology_set.card = 8)
  (h3 : (biology_set ∩ chemistry_set).card = 4)
  (h4 : ∀ p, p ∈ total_players → (p ∈ biology_set ∨ p ∈ chemistry_set)) :
  chemistry_set.card = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_chemistry_count_l338_33812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l338_33894

noncomputable def A : ℝ × ℝ := (-1, 3 * Real.sqrt 3)
noncomputable def B : ℝ × ℝ := (1, Real.sqrt 3)

noncomputable def circleIntersectsRay (r : ℝ) : ℝ × ℝ := 
  let a := -r/2
  (a, -Real.sqrt 3 * a)

noncomputable def circleIntersectsXAxis (r : ℝ) : ℝ × ℝ := (r, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum (r : ℝ) (hr : r > 0) :
  (∀ r' > 0, distance A (circleIntersectsRay r') + distance B (circleIntersectsXAxis r') 
    ≥ distance A (circleIntersectsRay r) + distance B (circleIntersectsXAxis r))
  → distance A (circleIntersectsRay r) + distance B (circleIntersectsXAxis r) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l338_33894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_parallelogram_on_strictly_convex_graph_l338_33844

-- Define a strictly convex function
def StrictlyConvex (f : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, x < t → t < y → f t < (y - t) / (y - x) * f x + (t - x) / (y - x) * f y

-- Theorem statement
theorem no_parallelogram_on_strictly_convex_graph (f : ℝ → ℝ) (h : StrictlyConvex f) :
  ¬ ∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧
    ((c - a, f c - f a) = (d - b, f d - f b) ∧
     (b - a, f b - f a) = (d - c, f d - f c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_parallelogram_on_strictly_convex_graph_l338_33844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l338_33827

/-- 
Given a cone whose lateral surface unfolds into a sector with a central angle of 120° and an area of 3π,
prove that the volume of the cone is (2√2/3)π.
-/
theorem cone_volume (l r h : ℝ) (h1 : l > 0) (h2 : r > 0) (h3 : h > 0) : 
  (3 * π = (1/3) * π * l^2) → 
  (2 * π / 3 = (r / l) * (2 * π)) →
  (h^2 = l^2 - r^2) →
  ((1/3) * π * r^2 * h = (2*Real.sqrt 2/3) * π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l338_33827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l338_33873

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- State that f is monotonically increasing
axiom f_monotone : Monotone f

-- State that f_inv is the inverse of f
axiom f_inv_def : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f

-- State that f(x+1) passes through points (-4, 0) and (2, 3)
axiom f_points : f (-3) = 0 ∧ f 3 = 3

-- State the condition |f^(-1)(x+1)| ≤ 3
axiom f_inv_bound : ∀ x, |f_inv (x + 1)| ≤ 3

-- Theorem to prove
theorem range_of_x : Set.Icc (-1 : ℝ) 2 = {x | ∃ y, f_inv (y + 1) = x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l338_33873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l338_33801

noncomputable def f (x : ℝ) (a : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x) ^ 2 + a

theorem f_properties (a : ℝ) :
  (∀ x, x ∈ Set.Icc (-π/6) (π/3) → f x a ≤ 3/2 ∧ f x a ≥ 0) →
  (∃ x y, x ∈ Set.Icc (-π/6) (π/3) ∧ y ∈ Set.Icc (-π/6) (π/3) ∧ f x a + f y a = 3/2) →
  (∀ x, f x 0 = Real.sin (2*x + π/6) + 1/2) ∧
  (∀ x, Real.sin x = f ((x/2) - π/12) 0 - 1/2) ∧
  (∫ x in (0)..(π/2), Real.sin x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l338_33801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l338_33821

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A line passing through the right focus of the hyperbola -/
structure FocalLine (h : Hyperbola) where
  slope : ℝ

/-- Predicate for a line intersecting both branches of the hyperbola at one point each -/
def intersects_both_branches_once (h : Hyperbola) (l : FocalLine h) : Prop :=
  l.slope = 1 → h.b / h.a > Real.sqrt 3 / 3

/-- Predicate for a line intersecting the right branch of the hyperbola at two distinct points -/
def intersects_right_branch_twice (h : Hyperbola) (l : FocalLine h) : Prop :=
  l.slope = 3 → h.b / h.a < Real.sqrt 3

theorem eccentricity_range (h : Hyperbola) 
  (l1 l2 : FocalLine h)
  (h1 : intersects_both_branches_once h l1)
  (h2 : intersects_right_branch_twice h l2) :
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l338_33821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_neg_quarter_f_three_roots_l338_33869

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -4 * x^2 else x^2 - x

-- Theorem for the values of a
theorem f_equals_neg_quarter (a : ℝ) : f a = -1/4 ↔ a = -1/4 ∨ a = 1/2 := by
  sorry

-- Theorem for the range of b
theorem f_three_roots (b : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = b ∧ f y = b ∧ f z = b) ↔ 
  -1/4 < b ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_neg_quarter_f_three_roots_l338_33869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l338_33862

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (1 : ℂ) / (1 - i) + i

theorem magnitude_of_z : Complex.abs z = Real.sqrt 10 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l338_33862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_bound_l338_33847

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x - 1)^2

-- State the theorem
theorem unique_zero_point_bound (a : ℝ) (x₀ : ℝ) (h_a : a > 0) 
  (h_x₀ : x₀ ∈ Set.Ioo 0 1) (h_unique : ∀ x ∈ Set.Ioo 0 1, f a x = 0 ↔ x = x₀) :
  Real.exp (-3/2) < x₀ ∧ x₀ < Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_bound_l338_33847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_f_plus_2x_nonnegative_iff_l338_33823

/-- The function f(x) = -1/a + 2/x for x > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_positive_iff (a : ℝ) (x : ℝ) (h : x > 0) :
  f a x > 0 ↔ (a > 0 ∧ 0 < x ∧ x < 2*a) ∨ (a < 0 ∧ x > 0) :=
sorry

theorem f_plus_2x_nonnegative_iff (a : ℝ) :
  (∀ x > 0, f a x + 2*x ≥ 0) ↔ a ∈ Set.Ioi (1/4) ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_f_plus_2x_nonnegative_iff_l338_33823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l338_33818

noncomputable def f (A w φ x : ℝ) := A * Real.sin (w * x + φ)

theorem sinusoidal_function_properties
  (A w φ : ℝ)
  (h_A : A > 0)
  (h_w : w > 0)
  (h_φ : |φ| < Real.pi / 2)
  (h_high : f A w φ (Real.pi / 6) = 2)
  (h_low : f A w φ (2 * Real.pi / 3) = -2) :
  ∃ (k : ℤ),
    (∀ x, f A w φ x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
    (∀ x, (∃ k : ℤ, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) ↔
      MonotoneOn (f A w φ) (Set.Icc x (x + Real.pi / 2))) ∧
    (∀ x, (∃ k : ℤ, x = k * Real.pi / 2 + Real.pi / 6) ↔
      ∀ y, f A w φ (x - y) = f A w φ (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l338_33818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_A_time_l338_33883

/-- Represents the time it takes for Machine A to complete the job alone -/
noncomputable def t : ℝ := sorry

/-- Represents the rate at which Machine A works -/
noncomputable def rate_A : ℝ := 1 / t

/-- Represents the rate at which Machine B works -/
noncomputable def rate_B : ℝ := rate_A / 4

/-- Represents the portion of the job completed by Machine A in 6 hours -/
noncomputable def portion_A : ℝ := 6 * rate_A

/-- Represents the portion of the job completed by Machine B in 8 hours -/
noncomputable def portion_B : ℝ := 8 * rate_B

theorem machine_A_time : t = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_A_time_l338_33883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_law_part_one_part_two_l338_33884

-- Define the distributive law
theorem distributive_law {α : Type*} [Ring α] (a b c : α) : a * (b + c) = a * b + a * c := by
  sorry

-- Part 1
theorem part_one : 5.6 * 4.32 + 5.6 * 5.68 = 56 := by
  sorry

-- Part 2
theorem part_two (a b m n : ℝ) (h1 : a + b = 3) (h2 : m + n = 4) : 
  a * m + a * n + b * m + b * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_law_part_one_part_two_l338_33884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_in_biology_l338_33824

/-- Given a college with 880 students where 50% are enrolled in biology classes,
    prove that 440 students are not enrolled in a biology class. -/
theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 1/2) :
  total_students - (biology_percentage * ↑total_students).floor = 440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_in_biology_l338_33824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_parabola_focus_locus_l338_33842

noncomputable section

/-- The fixed lower parabola -/
def fixed_parabola (x : ℝ) : ℝ := -x^2

/-- The moving upper parabola -/
def moving_parabola (x : ℝ) : ℝ := x^2

/-- The focus of the fixed parabola -/
def fixed_focus : ℝ × ℝ := (0, -1/4)

/-- The initial focus of the moving parabola -/
def initial_moving_focus : ℝ × ℝ := (0, 1/4)

/-- The point of tangency between the two parabolas -/
def tangency_point (l : ℝ) : ℝ × ℝ := (l, -l^2)

/-- The tangent line at the point of tangency -/
def tangent_line (l : ℝ) (x : ℝ) : ℝ := -2*l*x + l^2

/-- The locus of the focus of the moving parabola -/
def focus_locus (y : ℝ) : Prop := y = 1/4

theorem rolling_parabola_focus_locus :
  ∀ l : ℝ, ∃ x y : ℝ,
    (x, y) = (l, 1/4) ∧
    focus_locus y :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_parabola_focus_locus_l338_33842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l338_33875

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 (a, b > 0),
    foci F₁ and F₂, and points P and Q on the right branch such that
    PQ ⟂ PF₁ and |PF₁| = |PQ|, the eccentricity e of the hyperbola is √(5-2√2). -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) :
  a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ t : ℝ, P = F₂ + t • (Q - F₂)) →
  (P.1 - Q.1) * (P.2 - F₁.2) + (P.2 - Q.2) * (F₁.1 - P.1) = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) →
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = Real.sqrt (5 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l338_33875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l338_33820

/-- The natural exponential constant e -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) = a ln x - x² + x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + x

/-- The function g(x) = (x-2)e^x - x² + m -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x - x^2 + m

/-- The theorem stating the maximum value of positive integer m -/
theorem max_m_value : 
  ∃ (m : ℕ), m = 3 ∧ 
  (∀ (x : ℝ), x > 0 ∧ x ≤ 1 → f (-1) x > g m x) ∧
  (∀ (m' : ℕ), m' > m → ∃ (x : ℝ), x > 0 ∧ x ≤ 1 ∧ f (-1) x ≤ g m' x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l338_33820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_total_cost_l338_33831

/-- Calculates the total cost of Jessica's purchases after discount and including sales tax -/
def total_cost (cat_toy : ℚ) (cage : ℚ) (cat_food : ℚ) (leash : ℚ) (cat_treats : ℚ) 
                (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_discount := cat_toy + cage + cat_food + leash + cat_treats
  let discount := (cage * discount_rate).floor / 100 -- Rounding down to nearest cent
  let total_after_discount := total_before_discount - discount
  let sales_tax := (total_after_discount * tax_rate).floor / 100 -- Rounding down to nearest cent
  ((total_after_discount + sales_tax) * 100).floor / 100 -- Rounding down to nearest cent

/-- Theorem stating that Jessica's total cost is $40.03 -/
theorem jessica_total_cost : 
  total_cost 10.22 11.73 7.50 5.15 3.98 0.10 0.07 = 40.03 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_total_cost_l338_33831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_difference_l338_33852

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 9)
def B : ℝ × ℝ := (1, -1)
def C : ℝ × ℝ := (9, -1)

-- Define the vertical line that intersects AC at R and BC at S
noncomputable def R : ℝ × ℝ := sorry
noncomputable def S : ℝ × ℝ := sorry

-- State that R is on AC and S is on BC
axiom R_on_AC : (R.1 - A.1) / (C.1 - A.1) = (R.2 - A.2) / (C.2 - A.2)
axiom S_on_BC : (S.1 - B.1) / (C.1 - B.1) = (S.2 - B.2) / (C.2 - B.2)

-- State that RS is vertical
axiom RS_vertical : R.1 = S.1

-- Define the area of triangle RSC
def area_RSC : ℝ := 16

-- Theorem to prove
theorem coordinates_difference :
  |R.1 - R.2| = 10 - 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_difference_l338_33852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_72_l338_33840

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (fun d => d ∣ n ∧ d ≠ 0)

theorem sum_proper_divisors_72 :
  (proper_divisors 72).sum id = 123 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_72_l338_33840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l338_33825

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a ^ 2 + h.b ^ 2), 0)

/-- A point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x ^ 2 / h.a ^ 2 - y ^ 2 / h.b ^ 2 = 1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem hyperbola_eccentricity (h : Hyperbola) 
  (A : HyperbolaPoint h)
  (perp_asymptote : A.y / A.x = h.a / h.b)
  (dist_focus : distance (A.x, A.y) (right_focus h) = h.a / 2) :
  eccentricity h = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l338_33825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_determines_z_l338_33895

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

theorem projection_determines_z (z : ℝ) :
  let v : Fin 3 → ℝ := ![2, 4, z]
  let u : Fin 3 → ℝ := ![1, -2, 3]
  (dot_product v u / dot_product u u) • u = (10 / 14) • u →
  z = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_determines_z_l338_33895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l338_33816

noncomputable section

def rectangle_width : ℝ := 6
def rectangle_length : ℝ := 10

def cylinder1_height : ℝ := rectangle_length
def cylinder1_circumference : ℝ := rectangle_width
def cylinder2_height : ℝ := rectangle_width
def cylinder2_circumference : ℝ := rectangle_length

noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (circumference^2 * height) / (4 * Real.pi)

theorem cylinder_volume_ratio :
  let v1 := cylinder_volume cylinder1_height cylinder1_circumference
  let v2 := cylinder_volume cylinder2_height cylinder2_circumference
  max v1 v2 / min v1 v2 = 5/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l338_33816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_probabilities_l338_33891

def card_count : ℕ := 30

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_multiples_of_three (n : ℕ) : ℕ := 
  (List.range n).filter (fun x => (x + 1) % 3 = 0) |>.length

def count_odd_numbers (n : ℕ) : ℕ := 
  (List.range n).filter (fun x => (x + 1) % 2 = 1) |>.length

theorem card_probabilities :
  (count_multiples_of_three card_count : ℚ) / card_count = 1 / 3 ∧
  (count_odd_numbers card_count : ℚ) / card_count = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_probabilities_l338_33891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_l338_33876

-- Define the ellipse and its foci
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y - 2 = 0

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Define the tangent length from P(m, 0) to C
noncomputable def tangent_length (m : ℝ) : ℝ := Real.sqrt ((m - 2)^2 + 4 - 1)

theorem ellipse_circle_tangent :
  (∀ x y, ellipse x y ↔ x^2/4 + y^2/3 = 1) ∧
  left_focus = (-1, 0) ∧
  right_focus = (1, 0) ∧
  (∀ x y, symmetry_line x y ↔ x + y - 2 = 0) →
  (∀ x y, circle_C x y ↔ (x - 2)^2 + (y - 2)^2 = 1) ∧
  (∃ m, tangent_length m = Real.sqrt 3 ∧ 
        ∀ n, tangent_length n ≥ tangent_length m) ∧
  tangent_length 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_l338_33876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l338_33845

theorem cube_root_sum_equals_one :
  (2 + Real.sqrt 5) ^ (1/3 : ℝ) + (2 - Real.sqrt 5) ^ (1/3 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l338_33845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_tshirt_cost_l338_33859

/-- Represents the shop selling T-shirts -/
structure TShirtShop where
  womens_price : ℚ
  womens_interval : ℚ
  mens_interval : ℚ
  open_hours : ℚ
  weekly_earnings : ℚ

/-- Calculates the number of T-shirts sold per day -/
def shirts_per_day (interval : ℚ) (hours : ℚ) : ℚ :=
  (60 / interval) * hours

/-- Calculates the weekly sales of T-shirts -/
def weekly_sales (daily_sales : ℚ) : ℚ :=
  daily_sales * 7

/-- Theorem: The cost of a men's T-shirt is $15 -/
theorem mens_tshirt_cost (shop : TShirtShop) 
  (h1 : shop.womens_price = 18)
  (h2 : shop.womens_interval = 30)
  (h3 : shop.mens_interval = 40)
  (h4 : shop.open_hours = 12)
  (h5 : shop.weekly_earnings = 4914) :
  let womens_daily := shirts_per_day shop.womens_interval shop.open_hours
  let mens_daily := shirts_per_day shop.mens_interval shop.open_hours
  let womens_weekly := weekly_sales womens_daily
  let mens_weekly := weekly_sales mens_daily
  let womens_revenue := womens_weekly * shop.womens_price
  let mens_revenue := shop.weekly_earnings - womens_revenue
  mens_revenue / mens_weekly = 15 := by
  sorry

#eval shirts_per_day 30 12
#eval weekly_sales 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_tshirt_cost_l338_33859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_calculation_l338_33870

/-- Represents a cylindrical glass with lemonade --/
structure LemonadeGlass where
  height : ℝ
  diameter : ℝ
  fullness : ℝ
  lemonJuiceRatio : ℝ

/-- Calculates the volume of lemon juice in the glass --/
noncomputable def lemonJuiceVolume (glass : LemonadeGlass) : ℝ :=
  (Real.pi * (glass.diameter / 2)^2 * glass.height * glass.fullness) * 
  (glass.lemonJuiceRatio / (1 + glass.lemonJuiceRatio))

/-- Theorem stating the volume of lemon juice in the specified glass --/
theorem lemon_juice_volume_calculation :
  let glass : LemonadeGlass := {
    height := 8,
    diameter := 4,
    fullness := 1/3,
    lemonJuiceRatio := 1/5
  }
  lemonJuiceVolume glass = 16 * Real.pi / 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_calculation_l338_33870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_salt_concentration_l338_33880

/-- Represents a cup of salt water -/
structure SaltWater where
  total_weight : ℝ
  salt_weight : ℝ

/-- Calculates the salt concentration as a percentage -/
noncomputable def salt_concentration (sw : SaltWater) : ℝ :=
  (sw.salt_weight / sw.total_weight) * 100

/-- Theorem: Mixing two cups of salt water results in 22% concentration -/
theorem mixed_salt_concentration (cup1 cup2 : SaltWater)
  (h1 : cup1.total_weight = 200)
  (h2 : salt_concentration cup1 = 25)
  (h3 : cup2.total_weight = 300)
  (h4 : cup2.salt_weight = 60) :
  let mixed : SaltWater := {
    total_weight := cup1.total_weight + cup2.total_weight,
    salt_weight := cup1.salt_weight + cup2.salt_weight
  }
  salt_concentration mixed = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_salt_concentration_l338_33880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_point_equivalence_l338_33850

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Checks if a polar point is in standard representation -/
def isStandardPolar (p : PolarPoint) : Prop :=
  p.r > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi

/-- Defines equivalence of polar points -/
def polarEquivalent (p1 p2 : PolarPoint) : Prop :=
  p1.r * (Real.cos p1.θ) = p2.r * (Real.cos p2.θ) ∧
  p1.r * (Real.sin p1.θ) = p2.r * (Real.sin p2.θ)

theorem polar_point_equivalence :
  let p1 : PolarPoint := { r := -3, θ := Real.pi / 6 }
  let p2 : PolarPoint := { r := 3, θ := 7 * Real.pi / 6 }
  polarEquivalent p1 p2 ∧ isStandardPolar p2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_point_equivalence_l338_33850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_is_sqrt_3_l338_33886

/-- The number of match sticks available -/
def num_sticks : ℕ := 12

/-- The length of each match stick -/
def stick_length : ℝ := 1

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ

/-- A configuration of regular polygons made from match sticks -/
structure PolygonConfiguration where
  polygons : List RegularPolygon
  /-- The total number of sticks used must equal num_sticks -/
  total_sticks_used : (polygons.map (λ p => p.sides)).sum = num_sticks

/-- Calculates the area of a regular polygon -/
noncomputable def area (p : RegularPolygon) : ℝ := sorry

/-- Calculates the total area of a polygon configuration -/
noncomputable def total_area (config : PolygonConfiguration) : ℝ :=
  (config.polygons.map area).sum

/-- The smallest possible total area of polygons made from the match sticks -/
noncomputable def smallest_total_area : ℝ := Real.sqrt 3

theorem smallest_area_is_sqrt_3 :
  ∀ (config : PolygonConfiguration),
    total_area config ≥ smallest_total_area := by
  sorry

#check smallest_area_is_sqrt_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_is_sqrt_3_l338_33886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lemons_formula_l338_33841

-- Define the number of lemons for each person
noncomputable def levi_lemons (x : ℝ) : ℝ := x
noncomputable def jayden_lemons (x : ℝ) : ℝ := x + 6
noncomputable def alexandra_lemons (x : ℝ) : ℝ := (4/3) * (x + 6)
noncomputable def eli_lemons (x : ℝ) : ℝ := (2/3) * (x + 6)
noncomputable def ian_lemons (x : ℝ) : ℝ := (4/3) * (x + 6)
noncomputable def nathan_lemons (x : ℝ) : ℝ := (3/4) * x
noncomputable def olivia_lemons (x : ℝ) : ℝ := (4/5) * (x + 6)

-- Define the total number of lemons
noncomputable def total_lemons (x : ℝ) : ℝ :=
  levi_lemons x + jayden_lemons x + alexandra_lemons x + eli_lemons x +
  ian_lemons x + nathan_lemons x + olivia_lemons x

-- Theorem statement
theorem total_lemons_formula (x : ℝ) :
  total_lemons x = (413/60) * x + 30.8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lemons_formula_l338_33841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_inverse_minus_two_is_square_l338_33811

def a : ℕ → ℚ
  | 0 => 1/3  -- Added case for 0
  | 1 => 1/3
  | n + 2 => let a_nm2 := a n
              let a_nm1 := a (n + 1)
              ((1 - 2*a_nm2) * a_nm1^2) / (2*a_nm1^2 - 4*a_nm2*a_nm1^2 + a_nm2)

theorem a_inverse_minus_two_is_square (n : ℕ) :
  ∃ (k : ℤ), (1 / a n - 2 : ℚ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_inverse_minus_two_is_square_l338_33811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_value_l338_33881

def sequence_a (n : ℕ) : ℤ := 4 * n - 3

def sum_s (n : ℕ) : ℤ := 2 * n^2 - n

def sequence_b (n : ℕ) (k : ℤ) : ℚ := (sum_s n) / (n + k)

def sequence_t (n : ℕ) : ℚ := n / (2 * n + 1)

theorem minimum_m_value (a : ℕ → ℤ) (s : ℕ → ℤ) (b : ℕ → ℚ) (t : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n + 4) →
  (a 1 + a 4 = 14) →
  (∀ n, s n = sum_s n) →
  (∃ k : ℤ, ∀ n, b n = sequence_b n k) →
  (∃ d : ℚ, ∀ n, b (n + 1) - b n = d) →
  (∀ n, t n = sequence_t n) →
  (∀ n, t n ≤ 1 / 2) →
  (∀ m : ℕ, (∀ n, t n ≤ m / 100) → m ≥ 50) →
  ∃ m : ℕ, m = 50 ∧ (∀ n, t n ≤ m / 100) ∧ ∀ m' < m, ∃ n, t n > m' / 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_value_l338_33881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_symmetric_l338_33889

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1))

-- State the theorem
theorem f_increasing_and_symmetric :
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∃ a b, ∀ x, f (a + x) = f (a - x) + b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_symmetric_l338_33889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_12_factors_l338_33867

/-- The set of natural-number factors of 12 -/
def factors_of_12 : Finset ℕ := Finset.filter (λ n => 12 % n = 0) (Finset.range 13)

/-- The sum of reciprocals of the factors of 12 -/
def sum_of_reciprocals : ℚ := (factors_of_12.sum (λ n => (1 : ℚ) / n))

/-- Theorem stating that the sum of reciprocals of factors of 12 is 7/3 -/
theorem sum_of_reciprocals_of_12_factors :
  sum_of_reciprocals = 7/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_12_factors_l338_33867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l338_33822

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a train with length 180 meters crossing a pole in 9 seconds has a speed of 72 km/hr -/
theorem train_speed_calculation :
  train_speed 180 9 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l338_33822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l338_33838

/-- The equation of the trajectory of a point P(x, y) moving relative to an ellipse -/
theorem trajectory_equation (x y lambda : ℝ) (h_lambda : lambda ≠ 0) (h_x : x ≠ 1 ∧ x ≠ -1) :
  (∀ a b, a^2/4 + b^2/3 = 1 → (y/(x+1)) * (y/(x-1)) = lambda) →
  x^2 - y^2/lambda = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l338_33838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l338_33805

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

/-- The area of the region -/
noncomputable def region_area : ℝ := 16 * Real.pi

/-- Theorem stating the existence of a circle that matches the region equation and its area -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l338_33805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l338_33865

/-- The function h(t) = (t^2 + 0.5t) / (t^2 + 1) -/
noncomputable def h (t : ℝ) : ℝ := (t^2 + 0.5*t) / (t^2 + 1)

/-- The range of h is the singleton set {1/2} -/
theorem range_of_h : Set.range h = {1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l338_33865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_five_squares_l338_33804

/-- Definition of a square in our sequence -/
structure SquareS where
  i : ℕ
  side_length : ℝ
  area : ℝ

/-- The sequence of squares -/
noncomputable def square_sequence : ℕ → SquareS
  | 0 => ⟨0, 1, 1⟩
  | i + 1 => ⟨i + 1, (square_sequence i).side_length / 2, ((square_sequence i).side_length / 2)^2⟩

/-- The area of overlap between consecutive squares -/
noncomputable def overlap_area (i : ℕ) : ℝ :=
  (square_sequence i).area / 4

/-- The total area enclosed by at least one of the first n squares -/
noncomputable def total_enclosed_area (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => (square_sequence i).area) -
  (Finset.range (n-1)).sum (λ i => overlap_area i)

theorem enclosed_area_five_squares :
  total_enclosed_area 5 = 1279 / 1024 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_five_squares_l338_33804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_l338_33814

/-- The cost function for the Shanghai-Hangzhou Expressway trip -/
noncomputable def cost_function (v : ℝ) : ℝ := 166 * (0.02 * v + 200 / v)

/-- The domain of the cost function -/
def is_valid_speed (v : ℝ) : Prop := 60 ≤ v ∧ v ≤ 120

theorem minimum_cost :
  ∃ (v_min : ℝ), is_valid_speed v_min ∧
  (∀ (v : ℝ), is_valid_speed v → cost_function v_min ≤ cost_function v) ∧
  cost_function v_min = 664 ∧ v_min = 100 := by
  sorry

#check minimum_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_l338_33814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_area_calculation_l338_33888

noncomputable section

-- Define the circle and arcs
def circle_radius : ℝ := 1
def num_arcs : ℕ := 6
def arc_radius : ℝ := 1
def arc_angle : ℝ := 60 * Real.pi / 180  -- 60 degrees in radians

-- Define the area calculation functions
def sector_area (radius : ℝ) (angle : ℝ) : ℝ := (angle / (2 * Real.pi)) * Real.pi * radius^2

def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

def segment_area (radius : ℝ) (angle : ℝ) : ℝ :=
  sector_area radius angle - equilateral_triangle_area radius

-- Theorem statement
theorem black_area_calculation :
  let total_segment_area := num_arcs * segment_area circle_radius arc_angle
  let circle_area := Real.pi * circle_radius^2
  circle_area - total_segment_area = (3 * Real.sqrt 3) / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_area_calculation_l338_33888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l338_33829

def f (x : ℝ) : ℝ := |2*x - 2| + |x + 2|

theorem problem_solution :
  (∃ (T : ℝ),
    (∀ x, f x ≥ T) ∧
    (∃ x₀, f x₀ = T) ∧
    T = 3 ∧
    (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 + 2*b = T → a + b ≤ 2*Real.sqrt 2 - 1)) ∧
  {x : ℝ | f x ≤ 5 - 2*x} = Set.Icc (-5) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l338_33829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l338_33860

def solution_set (a : ℝ) : Set ℝ := {x : ℝ | |a * x - 2| < 6}

theorem inequality_solution_implies_a_value :
  ∀ a : ℝ, solution_set a = Set.Ioo (-1 : ℝ) 2 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l338_33860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_geometric_sequence_sine_l338_33899

noncomputable section

-- Define a right triangle with sides in geometric sequence
def RightTriangleGeometricSequence (a r : ℝ) : Prop :=
  a > 0 ∧ r > 0 ∧ a^2 + (a*r)^2 = (a*r^2)^2

-- Define the sine of the smallest angle
def SineSmallestAngle (a r : ℝ) : ℝ :=
  a / (a*r^2)

-- Define the condition for k not being divisible by the square of any prime
def NotDivisibleBySquarePrime (k : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ k)

-- Main theorem
theorem right_triangle_geometric_sequence_sine (a r : ℝ) (m n k : ℕ) :
  RightTriangleGeometricSequence a r →
  SineSmallestAngle a r = (m + Real.sqrt n : ℝ) / k →
  NotDivisibleBySquarePrime k →
  m + n + k = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_geometric_sequence_sine_l338_33899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_value_l338_33830

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  hexagons : ℕ
  vertices : ℕ
  edges : ℕ
  face_sum : faces = triangles + hexagons
  triangle_vertex : 3 * triangles = 5 * vertices
  hexagon_vertex : 2 * hexagons = 5 * vertices
  edge_formula : edges = (3 * triangles + 6 * hexagons) / 2
  euler_formula : vertices - edges + faces = 2

/-- The main theorem -/
theorem polyhedron_value (P : ConvexPolyhedron) 
  (h1 : P.faces = 60)
  (h2 : P.triangles = 20)
  (h3 : P.hexagons = 40) :
  100 * 2 + 10 * 3 + P.vertices = 322 := by
  have h4 : P.vertices = 92
  · -- Proof of vertex count
    sorry
  -- Main calculation
  calc
    100 * 2 + 10 * 3 + P.vertices
      = 200 + 30 + P.vertices := by ring
    _ = 200 + 30 + 92 := by rw [h4]
    _ = 322 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_value_l338_33830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l338_33898

theorem coefficient_x_cubed_in_expansion : 
  let n : ℕ := 5
  let a : ℚ := 2
  let expansion := (a - x)^n
  let coefficient_x_cubed := (Finset.range (n+1)).sum (λ k => 
    if k = 3 then (-1)^k * (n.choose k) * a^(n-k)
    else 0)
  coefficient_x_cubed = -40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l338_33898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_and_current_speed_log_drift_time_l338_33864

-- Define the problem parameters
variable (S : ℝ) -- distance between ports A and B
variable (a : ℝ) -- time from A to B downstream
variable (b : ℝ) -- time from B to A upstream
variable (x : ℝ) -- boat's speed in still water
variable (y : ℝ) -- current's speed

-- Theorem for part (2)①
theorem boat_and_current_speed :
  (x = (a + b) * S / (2 * a * b) ∧ y = (b - a) * S / (2 * a * b)) ↔
  (a * (x + y) = S ∧ b * (x - y) = S) :=
sorry

-- Theorem for part (2)②
theorem log_drift_time :
  (2 * a * b) / (b - a) = S / y :=
sorry

-- You can add more theorems or lemmas as needed

#check boat_and_current_speed
#check log_drift_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_and_current_speed_log_drift_time_l338_33864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptotes_l338_33828

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = (1/8) * y^2

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - (y^2/3) = 1

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The asymptotes of the hyperbola -/
def hyperbola_asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ :=
  abs (a * px + b * py + c) / Real.sqrt (a^2 + b^2)

theorem distance_focus_to_asymptotes :
  let (fx, fy) := parabola_focus
  ∀ x y, hyperbola_asymptotes x y →
    distance_point_to_line fx fy (Real.sqrt 3) (-1) 0 = Real.sqrt 3 ∧
    distance_point_to_line fx fy (Real.sqrt 3) 1 0 = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptotes_l338_33828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_inequality_l338_33854

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + b + 1

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := (f a b x) / x

theorem quadratic_function_and_inequality (a b : ℝ) (k : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Icc 2 3, f a b x ≤ 4) →
  (∃ x ∈ Set.Icc 2 3, f a b x = 4) →
  (∀ x ∈ Set.Icc 2 3, f a b x ≥ 1) →
  (∃ x ∈ Set.Icc 2 3, f a b x = 1) →
  (∀ x ∈ Set.Icc 1 2, g a b (2^x) - k * 2^x ≥ 0) →
  (f a b = fun x => x^2 - 2*x + 1) ∧ 
  (k ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_inequality_l338_33854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reuleaux_rotation_contact_length_l338_33887

/-- Represents a Reuleaux shape constructed from an equilateral triangle --/
structure ReuleauxShape where
  a : ℝ  -- Side length of the equilateral triangle
  r : ℝ  -- Radius of the small arcs

/-- The square frame in which the Reuleaux shape rotates --/
def squareFrame (R : ReuleauxShape) : ℝ := R.a + 2 * R.r

/-- The length of segments that the Reuleaux shape touches on the sides of the frame --/
noncomputable def touchedLength (R : ReuleauxShape) : ℝ := R.a * (3 - Real.sqrt 3) + 2 * R.r

/-- Theorem stating the relationship between the touched length and the frame size --/
theorem reuleaux_rotation_contact_length (R : ReuleauxShape) :
  touchedLength R = squareFrame R - R.a * (Real.sqrt 3 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reuleaux_rotation_contact_length_l338_33887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_10_plus_sqrt_41_l338_33855

noncomputable def A : ℝ × ℝ := (2, -3)
noncomputable def B : ℝ × ℝ := (8, 5)
noncomputable def C : ℝ × ℝ := (3, 9)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem total_distance_equals_10_plus_sqrt_41 :
  distance A B + distance B C = 10 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_10_plus_sqrt_41_l338_33855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_l338_33815

/-- Represents the weights of Leo, Kendra, and Ethan --/
structure Weights where
  leo : ℝ
  kendra : ℝ
  ethan : ℝ

/-- The conditions of the problem --/
def satisfies_conditions (w : Weights) : Prop :=
  w.leo + 10 = 1.5 * w.kendra ∧
  w.leo + 10 = 0.75 * w.ethan ∧
  w.leo + w.kendra + w.ethan = 210

/-- The theorem to be proved --/
theorem leo_weight (w : Weights) (h : satisfies_conditions w) :
  abs (w.leo - 63.33) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_l338_33815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_not_all_equal_geometric_mean_all_equal_l338_33813

/-- A type representing an infinite 2D grid of integers -/
def InfiniteGrid := ℤ → ℤ → ℤ

/-- A function to get the arithmetic mean of four numbers -/
def arithmeticMean (a b c d : ℤ) : ℚ := (a + b + c + d : ℚ) / 4

/-- A function to get the geometric mean of four numbers -/
noncomputable def geometricMean (a b c d : ℕ) : ℝ := ((a * b * c * d : ℝ) ^ (1/4 : ℝ))

/-- A predicate to check if a grid satisfies the arithmetic mean property -/
def satisfiesArithmeticMean (g : InfiniteGrid) : Prop :=
  ∀ x y : ℤ, g x y = ⌊arithmeticMean (g (x-1) y) (g (x+1) y) (g x (y-1)) (g x (y+1))⌋

/-- A predicate to check if a grid satisfies the geometric mean property -/
def satisfiesGeometricMean (g : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, (g x y : ℝ) = ⌊geometricMean (g (x-1) y) (g (x+1) y) (g x (y-1)) (g x (y+1))⌋

/-- Theorem: There exists a grid satisfying the arithmetic mean property but not all numbers are equal -/
theorem arithmetic_mean_not_all_equal :
  ∃ g : InfiniteGrid, satisfiesArithmeticMean g ∧ ¬(∀ x y x' y' : ℤ, g x y = g x' y') := by
  sorry

/-- Theorem: For any grid satisfying the geometric mean property, all numbers must be equal -/
theorem geometric_mean_all_equal :
  ∀ g : ℕ → ℕ → ℕ, satisfiesGeometricMean g → ∀ x y x' y' : ℕ, g x y = g x' y' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_not_all_equal_geometric_mean_all_equal_l338_33813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_formation_results_l338_33890

/-- Represents the number of members in each unit and the group size -/
structure GroupFormation where
  totalMembers : Nat
  unitAMembers : Nat
  unitBMembers : Nat
  groupSize : Nat

/-- Calculates the number of ways to form a group with exactly k members from unit B -/
def exactlyFromUnitB (g : GroupFormation) (k : Nat) : Nat :=
  Nat.choose g.unitBMembers k * Nat.choose g.unitAMembers (g.groupSize - k)

/-- Calculates the number of ways to form a group with at least k members from unit B -/
def atLeastFromUnitB (g : GroupFormation) (k : Nat) : Nat :=
  Finset.sum (Finset.range (g.unitBMembers - k + 1)) fun i => exactlyFromUnitB g (k + i)

/-- Calculates the number of ways to form a group excluding specific members -/
def excludingSpecificMembers (g : GroupFormation) : Nat :=
  Nat.choose g.totalMembers g.groupSize - Nat.choose (g.totalMembers - 2) (g.groupSize - 2)

/-- Theorem stating the results for the given problem -/
theorem group_formation_results (g : GroupFormation) 
    (h1 : g.totalMembers = 11) 
    (h2 : g.unitAMembers = 7) 
    (h3 : g.unitBMembers = 4) 
    (h4 : g.groupSize = 5) : 
  exactlyFromUnitB g 2 = 210 ∧ 
  atLeastFromUnitB g 2 = 301 ∧ 
  excludingSpecificMembers g = 378 := by
  sorry

#eval exactlyFromUnitB ⟨11, 7, 4, 5⟩ 2
#eval atLeastFromUnitB ⟨11, 7, 4, 5⟩ 2
#eval excludingSpecificMembers ⟨11, 7, 4, 5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_formation_results_l338_33890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_change_after_modification_l338_33877

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_change_after_modification :
  let original_area := triangle_area 15 20 18
  let new_area := triangle_area 30 26 18
  2 * original_area < new_area ∧ new_area < 3 * original_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_change_after_modification_l338_33877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l338_33810

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - abs x) + Real.log ((x^2 - 5*x + 6) / (x - 3))

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo 2 3 ∪ Set.Ioc 3 4

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | 4 - abs x ≥ 0 ∧ (x^2 - 5*x + 6) / (x - 3) > 0} = domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l338_33810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l338_33837

noncomputable def diamond (c d : ℝ) : ℝ := Real.sqrt (c^2 + d^2)

theorem diamond_calculation : 
  diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l338_33837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_group_size_l338_33857

theorem work_group_size (x : ℕ) 
  (h1 : (55 : ℝ) * x = (60 : ℝ) * (x - 15))
  (h2 : x > 0)
  (h3 : x - 15 > 0)
  : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_group_size_l338_33857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_three_l338_33858

theorem non_negative_integers_less_than_three : 
  {n : ℕ | n < 3} = Finset.range 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_three_l338_33858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_like_radicals_l338_33892

/-- Two radicals are like radicals if they have the same index and radicand -/
def like_radicals (x y : ℝ) (n : ℕ) : Prop :=
  x = y

/-- The simplest form of a radical expression -/
def simplest_radical_form (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ¬ ∃ (k : ℝ), k > 1 ∧ (∃ (n : ℕ), k^n = a ∧ k^n = b)

theorem unique_like_radicals :
  ∃! (a b : ℝ), a > 0 ∧ b > 0 ∧
    like_radicals (2 * a + b) 7 2 ∧
    like_radicals a 7 2 ∧
    simplest_radical_form (2 * a + b) 7 ∧
    simplest_radical_form a 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_like_radicals_l338_33892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_theorem_l338_33851

/-- Calculates the swimming speed in still water given the time taken to swim against a current, the distance covered, and the speed of the current. -/
noncomputable def swimming_speed_in_still_water (time : ℝ) (distance : ℝ) (current_speed : ℝ) : ℝ :=
  distance / time + current_speed

/-- Theorem stating that under the given conditions, the swimming speed in still water is 4 km/h. -/
theorem swimming_speed_theorem (time : ℝ) (distance : ℝ) (current_speed : ℝ) 
  (h1 : time = 7)
  (h2 : distance = 14)
  (h3 : current_speed = 2) :
  swimming_speed_in_still_water time distance current_speed = 4 := by
  sorry

#check swimming_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_theorem_l338_33851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_lcm_and_ratio_l338_33800

theorem gcd_of_lcm_and_ratio (C D : ℕ) : 
  Nat.lcm C D = 180 → C = 2 * (D / 5) → Nat.gcd C D = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_lcm_and_ratio_l338_33800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l338_33836

theorem relationship_abc : 
  let a := (0.3 : ℝ) ^ 2
  let b := Real.log 0.3 / Real.log 2
  let c := 2 ^ (0.3 : ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l338_33836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l338_33848

-- Define the lines
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + y - 1 = 0
def l₂ (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the angle of inclination
noncomputable def angle_of_inclination (a : ℝ) : ℝ := Real.arctan (-a)

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := a = 1

-- Define parallelism condition
def parallel (a : ℝ) : Prop := a = -1

-- Define distance between parallel lines
noncomputable def distance_between_parallel_lines (a : ℝ) : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem line_properties (a : ℝ) :
  (angle_of_inclination a = π / 4 → a = -1) ∧
  (perpendicular a → a = 1) ∧
  (parallel a → distance_between_parallel_lines a = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l338_33848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_after_mixing_and_evaporation_is_10_78_percent_l338_33803

/-- Calculates the percentage of alcohol in a solution after mixing and evaporation --/
noncomputable def alcohol_percentage_after_mixing_and_evaporation 
  (initial_volume : ℝ) 
  (initial_alcohol_percentage : ℝ) 
  (added_alcohol_volume : ℝ) 
  (added_water_volume : ℝ) 
  (evaporation_rate : ℝ) : ℝ :=
  let initial_alcohol_volume := initial_volume * initial_alcohol_percentage / 100
  let total_volume := initial_volume + added_alcohol_volume + added_water_volume
  let total_alcohol_before_evaporation := initial_alcohol_volume + added_alcohol_volume
  let evaporated_alcohol := total_alcohol_before_evaporation * evaporation_rate / 100
  let remaining_alcohol := total_alcohol_before_evaporation - evaporated_alcohol
  (remaining_alcohol / total_volume) * 100

/-- Theorem stating that the alcohol percentage after mixing and evaporation is approximately 10.78% --/
theorem alcohol_percentage_after_mixing_and_evaporation_is_10_78_percent :
  ∃ ε > 0, 
    ε < 0.01 ∧ 
    |alcohol_percentage_after_mixing_and_evaporation 40 5 3.5 6.5 2 - 10.78| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_after_mixing_and_evaporation_is_10_78_percent_l338_33803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_seven_fourths_l338_33807

open BigOperators Nat Real

noncomputable def f (n : ℕ) : ℝ := (totient (n^3))⁻¹

theorem fraction_equals_seven_fourths (m n : ℕ) :
  Nat.Coprime m n →
  (∑' k, f (2*k - 1)) / (∑' k, f (2*k)) = m / n →
  m / n = 7 / 4 :=
by sorry

#eval 100 * 7 + 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_seven_fourths_l338_33807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l338_33872

def my_sequence (n : ℕ+) : ℚ := n / (2*n - 1)

theorem sequence_formula (n : ℕ+) : 
  my_sequence n = n / (2*n - 1) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l338_33872
