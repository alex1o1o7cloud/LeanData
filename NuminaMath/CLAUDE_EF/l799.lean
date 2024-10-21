import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l799_79924

/-- The volume of a pyramid with a rectangular base and given dimensions -/
theorem pyramid_volume (base_length base_width height : ℝ) :
  base_length = 1 →
  base_width = 2 →
  height = 1 →
  (1/3 : ℝ) * base_length * base_width * height = 2/3 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l799_79924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_equation_correct_l799_79979

/-- Represents the time (in seconds) when Person A catches up with Person B in a running practice. -/
noncomputable def catch_up_time (speed_A speed_B head_start : ℝ) : ℝ :=
  head_start / (speed_A - speed_B)

/-- Theorem stating that the equation 7x - 5 = 6.5x correctly represents the catch-up time
    for the given running practice scenario. -/
theorem catch_up_equation_correct :
  let x := catch_up_time 7 6.5 5
  7 * x - 5 = 6.5 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_equation_correct_l799_79979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l799_79943

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)
variable (linePerpendicular : Line → Line → Prop)
variable (linePerpToPlane : Line → Plane → Prop)  -- New relation for line perpendicular to plane
variable (skew : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the lines and planes
variable (a b c : Line)
variable (α β γ : Plane)

theorem geometry_propositions :
  -- Proposition 1 (false)
  ¬(∀ α β γ : Plane, perpendicular α γ → perpendicular β γ → parallel α β) ∧
  -- Proposition 2 (true)
  (∀ a b : Line, ∀ α β : Plane,
    skew a b → contains α a → contains β b → lineParallelToPlane a β → lineParallelToPlane b α →
    parallel α β) ∧
  -- Proposition 3 (true)
  (∀ a b c : Line, ∀ α β γ : Plane,
    intersect α β a → intersect α γ b → intersect γ α c → lineParallel a b →
    lineParallelToPlane c β) ∧
  -- Proposition 4 (true)
  (∀ a b c : Line, ∀ α : Plane,
    skew a b → lineParallelToPlane a α → lineParallelToPlane b α → linePerpendicular c a → linePerpendicular c b →
    linePerpToPlane c α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l799_79943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_independent_of_height_enclosed_area_correct_l799_79988

/-- Represents the parameters of a projectile motion -/
structure ProjectileParams where
  h : ℝ  -- Initial height
  v : ℝ  -- Initial velocity
  g : ℝ  -- Acceleration due to gravity

/-- The x-coordinate of the highest point for a given angle -/
noncomputable def highest_point_x (p : ProjectileParams) (θ : ℝ) : ℝ :=
  (p.v^2 * Real.sin (2 * θ)) / (2 * p.g)

/-- The y-coordinate of the highest point for a given angle -/
noncomputable def highest_point_y (p : ProjectileParams) (θ : ℝ) : ℝ :=
  p.h + (p.v^2 * Real.sin θ^2) / (2 * p.g)

/-- The area enclosed by the curve traced by the highest points -/
noncomputable def enclosed_area (p : ProjectileParams) : ℝ :=
  (Real.pi / 8) * (p.v^4 / p.g^2)

/-- Theorem stating that the enclosed area is independent of the initial height -/
theorem enclosed_area_independent_of_height (p1 p2 : ProjectileParams) 
    (hv : p1.v = p2.v) (hg : p1.g = p2.g) : 
    enclosed_area p1 = enclosed_area p2 := by
  sorry

/-- Theorem stating that the enclosed area is correct -/
theorem enclosed_area_correct (p : ProjectileParams) : 
    ∃ (curve : ℝ → ℝ × ℝ), 
      (∀ θ, 0 ≤ θ ∧ θ ≤ Real.pi/2 → 
        curve θ = (highest_point_x p θ, highest_point_y p θ)) ∧
      (∃ A, A = enclosed_area p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_independent_of_height_enclosed_area_correct_l799_79988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_theorem_profit_after_discount_theorem_l799_79903

variable (a : ℝ)

/-- The selling price per unit given a cost and markup percentage -/
noncomputable def selling_price (cost : ℝ) (markup_percent : ℝ) : ℝ :=
  cost * (1 + markup_percent / 100)

/-- The discounted price given a selling price and discount percentage -/
noncomputable def discounted_price (price : ℝ) (discount_percent : ℝ) : ℝ :=
  price * (1 - discount_percent / 100)

/-- The profit per unit given a cost and selling price -/
noncomputable def profit_per_unit (cost : ℝ) (price : ℝ) : ℝ :=
  price - cost

theorem selling_price_theorem (a : ℝ) :
  selling_price a 22 = 1.22 * a := by sorry

theorem profit_after_discount_theorem (a : ℝ) :
  profit_per_unit a (discounted_price (selling_price a 22) 15) = 0.037 * a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_theorem_profit_after_discount_theorem_l799_79903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l799_79940

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2) * Real.sin (5 * Real.pi / 2 + α)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) :
  f α = (1/2) * Real.cos (2 * α) + 1/2 := by sorry

theorem f_specific_value (α : Real) (h : Real.cos (5 * Real.pi / 6 + 2 * α) = 1/3) :
  f (Real.pi / 12 - α) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l799_79940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l799_79913

-- Define the natural numbers without zero
def PositiveNat := { n : ℕ // n > 0 }

-- Define the sequences
def a : PositiveNat → ℝ := sorry
def b : PositiveNat → ℝ := sorry
def c : PositiveNat → ℝ := sorry

-- Define the sum of the first n terms of sequence a
def S (n : PositiveNat) : ℝ := n.val ^ 2

-- Define the sum of the first n terms of sequence c
def T : PositiveNat → ℝ := sorry

-- State the theorem
theorem sequence_sum_theorem (n : PositiveNat) :
  -- Conditions
  (∀ k : PositiveNat, S k = k.val ^ 2) →
  (b ⟨1, by norm_num⟩ = a ⟨1, by norm_num⟩) →
  (2 * b ⟨3, by norm_num⟩ = b ⟨4, by norm_num⟩) →
  (∀ k : PositiveNat, c k = a k * b k) →
  -- Conclusion
  T n = (2 * n.val - 3) * 2^n.val + 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l799_79913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_l799_79959

/-- Represents the scale of a map --/
structure MapScale where
  inches : ℚ
  miles : ℚ

/-- Calculates the actual distance between two cities given a map distance and scale --/
def actualDistance (mapDistance : ℚ) (scale : MapScale) : ℚ :=
  (mapDistance * scale.miles) / scale.inches

/-- Theorem stating the actual distance between two cities --/
theorem city_distance (mapDistance : ℚ) (scale : MapScale) 
  (h1 : mapDistance = 20)
  (h2 : scale.inches = 2/5)  -- 0.4 as a rational number
  (h3 : scale.miles = 5) :
  actualDistance mapDistance scale = 250 := by
  sorry

#eval actualDistance 20 { inches := 2/5, miles := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_l799_79959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quincy_car_price_l799_79997

/-- Represents the price of a car purchased with a loan -/
def car_price (down_payment : ℝ) (monthly_payment : ℝ) (loan_duration_months : ℕ) : ℝ :=
  down_payment + monthly_payment * (loan_duration_months : ℝ)

/-- Theorem: The price of Quincy's car is $20,000 -/
theorem quincy_car_price :
  let down_payment : ℝ := 5000
  let monthly_payment : ℝ := 250
  let loan_duration_months : ℕ := 60  -- 5 years * 12 months
  car_price down_payment monthly_payment loan_duration_months = 20000 := by
  sorry

#check quincy_car_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quincy_car_price_l799_79997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l799_79983

noncomputable def f (x : ℝ) := x * Real.sin x

theorem properties_of_f :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x : ℝ, |f x| ≤ |x|) ∧
  (∀ k : ℝ, |k| > 1 → (∃! x : ℝ, f x = k * x)) :=
by
  sorry

#check properties_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l799_79983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_do_not_hold_l799_79996

noncomputable def hash (a b : ℝ) : ℝ := 3 * (a + b) / 2

theorem distributive_laws_do_not_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_do_not_hold_l799_79996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_probability_l799_79909

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of games in the series -/
def n_games : ℕ := 7

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The probability of the Lakers winning the NBA finals in seven games -/
def p_lakers_win_series : ℚ := 135/4096

theorem lakers_win_probability :
  p_lakers_win_series = (Nat.choose 6 3 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_probability_l799_79909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_stripe_probability_is_three_sixteenths_l799_79926

/-- A rectangular prism with painted stripes on each face -/
structure StripedPrism where
  /-- Total number of faces on the rectangular prism -/
  num_faces : Nat
  /-- Number of possible stripe orientations for each face -/
  stripe_orientations : Nat
  /-- Number of faces required to form a continuous stripe circuit -/
  circuit_faces : Nat

/-- Calculate the total number of possible stripe combinations -/
def total_combinations (prism : StripedPrism) : Nat :=
  prism.stripe_orientations ^ prism.num_faces

/-- Calculate the number of stripe combinations that form a continuous circuit -/
def circuit_combinations (prism : StripedPrism) : Nat :=
  (prism.num_faces / 2) * (prism.stripe_orientations ^ 2)

/-- Calculate the probability of a continuous stripe encircling the prism -/
def continuous_stripe_probability (prism : StripedPrism) : ℚ :=
  ↑(circuit_combinations prism) / ↑(total_combinations prism)

/-- Theorem: The probability of a continuous stripe encircling a rectangular prism is 3/16 -/
theorem continuous_stripe_probability_is_three_sixteenths :
  continuous_stripe_probability { num_faces := 6, stripe_orientations := 2, circuit_faces := 4 } = 3 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_stripe_probability_is_three_sixteenths_l799_79926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_cube_l799_79929

theorem cube_root_of_cube (π : ℝ) : Real.sqrt ((π - 2)^3) = π - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_cube_l799_79929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_lent_years_l799_79942

-- Define the principal amount
def P : ℝ := 453.51473922902494

-- Define the final amount
def A : ℝ := 500

-- Define the annual interest rate
def r : ℝ := 0.05

-- Define the number of times interest is compounded per year
def n : ℝ := 1

-- Define the function to calculate the number of years
noncomputable def calculateYears (P A r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

-- Theorem statement
theorem money_lent_years : 
  ⌊calculateYears P A r n⌋ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_lent_years_l799_79942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l799_79982

theorem parity_of_expression (a b c : ℕ) (ha : Odd a) (hb : Even b) :
  Odd (3^a + (b+1)^2*c) ↔ Even c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l799_79982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_l799_79960

theorem coin_distribution (coins : Fin 77 → ℕ) 
  (h2 : ∀ (i j : Fin 77), i ≠ j → ∃ k : ℕ, coins i + coins j = 2 * k)
  (h_general : ∀ (n : ℕ) (s : Finset (Fin 77)), 3 ≤ n → n ≤ 76 → s.card = n → 
    ∃ k : ℕ, (s.sum (λ i => coins i)) = n * k) :
  ∃ k : ℕ, (Finset.univ.sum (λ i => coins i)) = 77 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_l799_79960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_of_equilateral_triangular_prism_l799_79994

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents an equilateral triangular prism -/
structure EquilateralTriangularPrism where
  S : Point3D  -- apex
  A : Point3D  -- base vertex
  B : Point3D  -- base vertex
  C : Point3D  -- base vertex
  height : ℝ   -- height of the prism
  baseEdge : ℝ -- length of base edge

/-- Helper function to represent the plane SBC -/
def plane_SBC (prism : EquilateralTriangularPrism) : Set Point3D :=
  sorry

/-- Helper function to represent the base plane -/
def base_plane (prism : EquilateralTriangularPrism) : Set Point3D :=
  sorry

/-- Helper function to check if a line is perpendicular to a plane -/
def is_perpendicular (A : Point3D) (O' : Point3D) (plane : Set Point3D) : Prop :=
  sorry

/-- Helper function to calculate the ratio of two segments -/
def segment_ratio (A : Point3D) (P : Point3D) (O' : Point3D) : ℝ :=
  sorry

/-- Helper function to calculate the area of a cross-section -/
def area_of_cross_section (P : Point3D) (base : Set Point3D) : ℝ :=
  sorry

/-- The theorem statement -/
theorem cross_section_area_of_equilateral_triangular_prism 
  (prism : EquilateralTriangularPrism)
  (O' : Point3D)  -- foot of perpendicular from A to SBC
  (P : Point3D)   -- point on AO'
  (h1 : prism.height = 3)
  (h2 : prism.baseEdge = 6)
  (h3 : is_perpendicular prism.A O' (plane_SBC prism))
  (h4 : segment_ratio prism.A P O' = 8)
  : area_of_cross_section P (base_plane prism) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_of_equilateral_triangular_prism_l799_79994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_four_times_sqrt_two_l799_79949

theorem cube_root_four_times_sqrt_two : 
  (4 : ℝ) ^ (1/3 : ℝ) * (2 : ℝ) ^ (1/2 : ℝ) = 2 ^ (7/6 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_four_times_sqrt_two_l799_79949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l799_79915

/-- Calculates the market value of a stock given its dividend rate, yield, and face value. -/
noncomputable def market_value (dividend_rate : ℝ) (yield : ℝ) (face_value : ℝ) : ℝ :=
  (dividend_rate * face_value / yield) * 100

/-- Theorem stating that a 6% stock yielding 8% with a face value of $100 has a market value of $75. -/
theorem stock_market_value :
  let dividend_rate : ℝ := 0.06
  let yield : ℝ := 0.08
  let face_value : ℝ := 100
  market_value dividend_rate yield face_value = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l799_79915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_n_l799_79900

/-- Sum of first k terms of an arithmetic progression -/
noncomputable def sum_ap (a d k : ℝ) : ℝ := k / 2 * (2 * a + (k - 1) * d)

/-- Definition of R in terms of sums of arithmetic progression -/
noncomputable def R (a d n : ℝ) : ℝ :=
  sum_ap a d (3 * n) - sum_ap a d (2 * n) - sum_ap a d n

/-- Theorem stating that R depends only on d and n -/
theorem R_depends_on_d_and_n (a d n : ℝ) :
  R a d n = 2 * d * n^2 := by
  -- Expand the definition of R
  unfold R
  -- Expand the definition of sum_ap
  unfold sum_ap
  -- Simplify the algebraic expression
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_n_l799_79900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l799_79937

theorem trigonometric_identities (x : Real) 
  (h1 : -π/2 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ (Real.tan x = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l799_79937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_in_range_l799_79956

theorem unique_solution_in_range : ∃! x : ℝ, 
  (Real.sqrt (x + 15) - 3 / Real.sqrt (x + 15) = 4) ∧ 
  (30 < x) ∧ (x < 40) ∧ 
  (x = 11 + 4 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_in_range_l799_79956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_reach_target_l799_79906

noncomputable def initial_height : ℝ := 360
noncomputable def bounce_fraction : ℝ := 3/4
noncomputable def target_height : ℝ := 40

noncomputable def height_after_bounces (n : ℕ) : ℝ :=
  initial_height * (bounce_fraction ^ n)

def reaches_target (n : ℕ) : Prop :=
  height_after_bounces n < target_height

theorem min_bounces_to_reach_target :
  ∃ n : ℕ, reaches_target n ∧ ∀ m : ℕ, m < n → ¬reaches_target m :=
by sorry

#eval 8 -- The expected answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_reach_target_l799_79906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l799_79912

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- The angle between two lines in radians -/
noncomputable def angle_between (l1 l2 : OriginLine) : ℝ := sorry

/-- The line y = √3x + 2 -/
noncomputable def reference_line : OriginLine := ⟨Real.sqrt 3⟩

theorem line_equation (l : OriginLine) :
  angle_between l reference_line = π / 6 →
  (l.slope = 0 ∨ l.slope = Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l799_79912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_identification_probability_l799_79957

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

noncomputable def derangement (n : ℕ) : ℚ := 
  (List.range (n + 1)).foldl (fun acc i => acc + ((-1 : ℚ)^i * (factorial n : ℚ) / (factorial i : ℚ))) 0

theorem baby_identification_probability : 
  let total_babies : ℕ := 8
  let correct_identifications : ℕ := 4
  let total_permutations := factorial total_babies
  let ways_to_choose_correct := choose total_babies correct_identifications
  let ways_to_derange_rest := derangement (total_babies - correct_identifications)
  (ways_to_choose_correct : ℚ) * ways_to_derange_rest / total_permutations = 1 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_identification_probability_l799_79957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_white_possible_l799_79971

def N : ℕ := 1000000

structure Move where
  number : Fin N
  deriving Repr

def applyMove (colors : Fin N → Bool) (m : Move) : Fin N → Bool :=
  λ n => if ¬(Nat.gcd n.val (m.number.val + 1) = 1) then ¬(colors n) else colors n

def initialColors : Fin N → Bool :=
  λ _ => false  -- All black initially

theorem all_white_possible :
  ∃ (moves : List Move), (moves.foldl applyMove initialColors) = λ _ => true :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_white_possible_l799_79971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l799_79917

-- Define the vertices of the square
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (1, 7)
def C : ℝ × ℝ := (-3, 6)
def D : ℝ × ℝ := (-2, 2)

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of the quadrilateral
noncomputable def area : ℝ := distance A B * distance A B

-- Theorem stating that the area of the quadrilateral is 17
theorem square_area : area = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l799_79917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_max_n_value_l799_79972

noncomputable def a (n : ℕ) : ℚ := 3 * (1/2)^n

noncomputable def S (n : ℕ) : ℚ := 3 * (1 - (1/2)^n)

theorem sequence_sum_property (n : ℕ) :
  2 * a (n + 1) + S n = 3 :=
sorry

theorem max_n_value :
  ∀ n : ℕ, (n ≥ 1 ∧ S (2*n) / S n > 64/63) → n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_max_n_value_l799_79972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_order_l799_79941

theorem root_order : (729 : ℝ) ^ (1/3) < Real.sqrt 121 ∧ Real.sqrt 121 < (38416 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_order_l799_79941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_sqrt3_div_3_l799_79995

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 2^(x - 1) else Real.tan (Real.pi * x / 3)

theorem f_composition_equals_sqrt3_div_3 :
  f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_sqrt3_div_3_l799_79995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l799_79935

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2*(x - 1)

-- Define the right focus of hyperbola C
def right_focus_C : ℝ × ℝ := (2, 0)

-- Define the focus of parabola E
def focus_E : ℝ × ℝ := (2, 0)

-- Define the right vertex of hyperbola C
def right_vertex_C : ℝ × ℝ := (1, 0)

-- Define points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- Theorem statement
theorem hyperbola_parabola_intersection :
  (∀ x y, hyperbola_C x y → parabola_E x y → line_l x y → (x, y) = M ∨ (x, y) = N) →
  right_focus_C = focus_E →
  line_l (right_vertex_C.1) (right_vertex_C.2) →
  (∀ x y, parabola_E x y ↔ y^2 = 8*x) ∧
  ∃ d, d = ‖M - N‖ ∧ d^2 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l799_79935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_of_polynomial_l799_79932

theorem factor_of_polynomial : 
  ∃ q : Polynomial ℝ, X^4 + 16 = (X^2 - 4*X + 4) * q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_of_polynomial_l799_79932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_side_length_of_box_l799_79923

-- Define the internal volume in cubic feet
def internal_volume : ℝ := 4

-- Define the wall thickness in inches
def wall_thickness : ℝ := 1

-- Define the conversion factor from cubic feet to cubic inches
def cubic_feet_to_cubic_inches : ℝ := 1728

-- Theorem statement
theorem external_side_length_of_box :
  ∀ (internal_side_length : ℝ),
    internal_side_length > 0 →
    internal_side_length^3 = internal_volume * cubic_feet_to_cubic_inches →
    ∃ (external_side_length : ℝ),
      external_side_length = internal_side_length + 2 * wall_thickness ∧
      abs (external_side_length - 21.08) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_side_length_of_box_l799_79923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_four_l799_79948

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Line in polar coordinates -/
noncomputable def line (θ : ℝ) : ℝ := 5 * Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

/-- Angle of ray OM -/
noncomputable def θ_ray : ℝ := Real.pi / 3

/-- Point P: Intersection of ray OM and curve C -/
noncomputable def P : ℝ := curve_C θ_ray

/-- Point Q: Intersection of ray OM and the line -/
noncomputable def Q : ℝ := line θ_ray

theorem length_PQ_is_four : abs (Q - P) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_four_l799_79948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equality_l799_79930

theorem cos_shift_equality (x : ℝ) : 
  Real.cos (2 * (x - Real.pi / 4)) = 2 * Real.cos x * Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equality_l799_79930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_asymptote_l799_79904

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ := (3, 0)

-- Define a point M
variable (M : ℝ × ℝ)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := 4 * x - 3 * y = 0

-- Theorem statement
theorem min_distance_to_asymptote :
  hyperbola P.1 P.2 →
  (M.1^2 + M.2^2 = 1) →
  (P.1 - O.1) * (M.1 - O.1) + (P.2 - O.2) * (M.2 - O.2) = 0 →
  (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → 
    (Q.1 - M.1)^2 + (Q.2 - M.2)^2 ≥ (P.1 - M.1)^2 + (P.2 - M.2)^2) →
  (|4 * P.1 - 3 * P.2| / Real.sqrt (4^2 + (-3)^2) = 12 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_asymptote_l799_79904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_impossible_l799_79973

/-- Represents a 3x3x3 cube configuration --/
def Cube := Fin 3 → Fin 3 → Fin 3 → Bool

/-- Counts the number of true values in a line of 3 --/
def countLine (line : Fin 3 → Bool) : Nat :=
  (line 0).toNat + (line 1).toNat + (line 2).toNat

/-- Checks if a line has an odd count --/
def isLineOdd (line : Fin 3 → Bool) : Bool :=
  countLine line % 2 = 1

/-- Checks if all lines in the cube have an odd count --/
def allLinesOdd (cube : Cube) : Prop :=
  (∀ i j, isLineOdd (λ k => cube i j k)) ∧
  (∀ i k, isLineOdd (λ j => cube i j k)) ∧
  (∀ j k, isLineOdd (λ i => cube i j k))

/-- Counts the total number of true values in the cube --/
def countCube (cube : Cube) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j =>
      Finset.sum (Finset.univ : Finset (Fin 3)) (λ k =>
        (cube i j k).toNat)))

/-- The main theorem stating the impossibility of the arrangement --/
theorem cube_arrangement_impossible :
  ¬ ∃ (cube : Cube), allLinesOdd cube ∧ countCube cube = 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_impossible_l799_79973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_correct_f_well_defined_on_domain_l799_79905

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - Real.sqrt (4 - Real.sqrt (5 - x)))

-- Define the domain of f
def domain_of_f : Set ℝ := { x | -11 ≤ x ∧ x ≤ 5 }

-- Theorem stating that the domain of f is [-11, 5]
theorem domain_of_f_is_correct :
  ∀ x : ℝ, x ∈ domain_of_f ↔ (5 - x ≥ 0 ∧ 4 - Real.sqrt (5 - x) ≥ 0 ∧ 2 - Real.sqrt (4 - Real.sqrt (5 - x)) ≥ 0) :=
by sorry

-- Theorem stating that f is well-defined on its domain
theorem f_well_defined_on_domain :
  ∀ x ∈ domain_of_f, (5 - x ≥ 0 ∧ 4 - Real.sqrt (5 - x) ≥ 0 ∧ 2 - Real.sqrt (4 - Real.sqrt (5 - x)) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_correct_f_well_defined_on_domain_l799_79905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l799_79958

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 39, 32, and 10 is approximately 129.35 -/
theorem triangle_area_example : 
  ∃ ε > 0, |triangle_area 39 32 10 - 129.35| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l799_79958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l799_79964

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  first_term : ℚ
  seq_property : ∀ n, a (n + 1) = a n + d
  first_term_prop : a 0 = first_term

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first_term + (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : sum_n_terms seq 3 / 3 - sum_n_terms seq 2 / 2 = 1) :
  seq.d = 2 := by
  sorry

#check arithmetic_sequence_common_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l799_79964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_with_distance_3_from_focus_l799_79986

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_points_with_distance_3_from_focus :
  ∀ x y : ℝ, parabola x y ∧ distance (x, y) focus = 3 ↔ 
  (x = 2 ∧ y = 2 * Real.sqrt 2) ∨ (x = 2 ∧ y = -2 * Real.sqrt 2) := by
  sorry

#check parabola_points_with_distance_3_from_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_with_distance_3_from_focus_l799_79986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_pack_ranking_l799_79922

/-- Represents a pack of cookies -/
structure Pack where
  size : String
  cost : ℚ
  quantity : ℚ

/-- Calculates the cost per cookie for a given pack -/
def costPerCookie (p : Pack) : ℚ := p.cost / p.quantity

theorem cookie_pack_ranking (s m l : Pack) 
  (h1 : s.size = "S" ∧ m.size = "M" ∧ l.size = "L")
  (h2 : m.cost = 14/10 * s.cost)
  (h3 : m.quantity = 3/4 * l.quantity)
  (h4 : l.quantity = 3 * s.quantity)
  (h5 : l.cost = 5/4 * m.cost) :
  costPerCookie l < costPerCookie m ∧ costPerCookie m < costPerCookie s :=
by
  sorry

#eval "Cookie pack ranking theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_pack_ranking_l799_79922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_chairs_cost_l799_79901

/-- Calculates the total cost of chairs with a discount structure -/
noncomputable def totalCost (normalPrice : ℝ) (numChairs : ℕ) : ℝ :=
  let initialDiscount := 0.25 * normalPrice
  let discountedPrice := normalPrice - initialDiscount
  let firstFiveCost := min numChairs 5 * discountedPrice
  let additionalChairs := max (numChairs - 5) 0
  let additionalDiscount := discountedPrice * (1/3)
  let additionalCost := additionalChairs * (discountedPrice - additionalDiscount)
  firstFiveCost + additionalCost

/-- The theorem stating the total cost of 8 chairs -/
theorem eight_chairs_cost (normalPrice : ℝ) (h : normalPrice = 20) :
  totalCost normalPrice 8 = 105 := by
  sorry

-- Note: We can't use #eval with noncomputable functions, so we'll remove this line
-- #eval totalCost 20 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_chairs_cost_l799_79901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_problem_l799_79984

-- Define the transformation rule
def transform (a b c : ℤ) : ℤ × ℤ × ℤ := (b, c, a + b - 1)

-- Define a function to iterate the transformation n times
def iterate_transform : ℕ → ℤ × ℤ × ℤ → ℤ × ℤ × ℤ
  | 0, state => state
  | n + 1, (a, b, c) => iterate_transform n (transform a b c)

-- Define a predicate to check if a triple can be transformed to the final state
def can_reach_final_state (initial : ℤ × ℤ × ℤ) : Prop :=
  ∃ (n : ℕ), 
    (iterate_transform n initial = (17, 1967, 1983)) ∨
    (iterate_transform n initial = (17, 1983, 1967)) ∨
    (iterate_transform n initial = (1967, 17, 1983)) ∨
    (iterate_transform n initial = (1967, 1983, 17)) ∨
    (iterate_transform n initial = (1983, 17, 1967)) ∨
    (iterate_transform n initial = (1983, 1967, 17))

-- The theorem to be proved
theorem blackboard_problem :
  can_reach_final_state (3, 3, 3) ∧ ¬can_reach_final_state (2, 2, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_problem_l799_79984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l799_79970

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_function_property
  (a b c : ℝ)
  (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ x, (quadratic_function a b c x - x) * (quadratic_function a b c x - (x^2 + 1) / 2) ≤ 0) :
  quadratic_function a b c 1 = 1 ∧ a = 1/4 ∧ b = 1/2 ∧ c = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l799_79970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_root_l799_79963

/-- The function f(x) = x^x + x^(1-x) - x - 1 --/
noncomputable def f (x : ℝ) : ℝ := x^x + x^(1-x) - x - 1

/-- Theorem stating that x = 1 is the only positive root of x^x + x^(1-x) = x + 1 --/
theorem unique_positive_root :
  ∀ x : ℝ, x > 0 → (f x = 0 ↔ x = 1) := by
  sorry

#check unique_positive_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_root_l799_79963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_satisfying_inequality_l799_79991

theorem positive_integers_satisfying_inequality : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ (130 * n)^50 > n^100 ∧ n^100 > 5^250) (Finset.range 130)).card = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_satisfying_inequality_l799_79991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_in_quarter_A_l799_79914

/-- Represents the quarters of the circular track -/
inductive Quarter
| A
| B
| C
| D

/-- Determines the quarter in which a runner stops after running a given distance on a circular track -/
def stopQuarter (trackCircumference : ℕ) (runDistance : ℕ) : Quarter :=
  match runDistance % trackCircumference with
  | 0 => Quarter.A
  | n => 
    if n ≤ trackCircumference / 4 then Quarter.A
    else if n ≤ trackCircumference / 2 then Quarter.B
    else if n ≤ 3 * trackCircumference / 4 then Quarter.C
    else Quarter.D

/-- Theorem stating that running 8000 feet on a 40-foot circular track results in stopping in quarter A -/
theorem runner_stops_in_quarter_A :
  stopQuarter 40 8000 = Quarter.A := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_in_quarter_A_l799_79914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_exists_A₄_l799_79978

-- Define the points A₁, A₂, and A₃
def A₁ : ℝ × ℝ := (1, 0)
def A₂ : ℝ × ℝ := (-2, 0)
def A₃ : ℝ × ℝ := (-1, 0)

-- Define the ratio lambda
noncomputable def lambda : ℝ := Real.sqrt 2 / 2

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Part 1: Locus of point M
theorem locus_of_M (x y : ℝ) :
  distance (x, y) A₁ / distance (x, y) A₂ = lambda → x^2 + y^2 - 8*x - 2 = 0 :=
by sorry

-- Part 2: Existence of point A₄
theorem exists_A₄ :
  ∃ (m n : ℝ), ∀ (x y : ℝ),
    (x - 3)^2 + y^2 = 4 →
    distance (x, y) A₃ / distance (x, y) (m, n) = 2 ∧ m = 2 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_exists_A₄_l799_79978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_points_form_line_segment_l799_79944

/-- Two distinct points in a metric space -/
structure TwoPoints (X : Type*) [MetricSpace X] where
  F₁ : X
  F₂ : X
  distinct : F₁ ≠ F₂

/-- The set of points M such that |MF₁| + |MF₂| = |F₁F₂| -/
def ConstantSumPoints (X : Type*) [MetricSpace X] (points : TwoPoints X) : Set X :=
  {M : X | dist M points.F₁ + dist M points.F₂ = dist points.F₁ points.F₂}

/-- The line segment between two points -/
def LineSegment (X : Type*) [MetricSpace X] [NormedAddCommGroup X] [InnerProductSpace ℝ X] (A B : X) : Set X :=
  {M : X | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B}

theorem constant_sum_points_form_line_segment
  (X : Type*) [MetricSpace X] [NormedAddCommGroup X] [InnerProductSpace ℝ X]
  (points : TwoPoints X)
  (h : dist points.F₁ points.F₂ = 16) :
  ConstantSumPoints X points = LineSegment X points.F₁ points.F₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_points_form_line_segment_l799_79944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_range_l799_79910

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 2 - 3 * a else 2^x - 1

theorem function_equality_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) → a ∈ Set.Iio (2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_range_l799_79910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_symmetric_triangle_l799_79985

-- Define a right triangle XYZ
structure RightTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  is_right : (X.1 - Z.1) * (Y.1 - Z.1) + (X.2 - Z.2) * (Y.2 - Z.2) = 0

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Define symmetric point with respect to a line
noncomputable def symmetric_point (P A B : ℝ × ℝ) : ℝ × ℝ :=
  (2 * (A.1 + B.1) / 2 - P.1, 2 * (A.2 + B.2) / 2 - P.2)

theorem area_of_symmetric_triangle (t : RightTriangle) :
  triangle_area t.X t.Y t.Z = 1 →
  triangle_area 
    (symmetric_point t.X t.Y t.Z)
    (symmetric_point t.Y t.X t.Z)
    (symmetric_point t.Z t.X t.Y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_symmetric_triangle_l799_79985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_minus_one_makes_f_linear_l799_79969

/-- A function f is linear if there exist constants k ≠ 0 and b such that f(x) = k * x + b for all x. -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x + b

/-- The function y = (m-1)x^(m^2) + 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^(m^2) + 1

/-- Theorem: The only value of m that makes f(m) a linear function is -1 -/
theorem only_minus_one_makes_f_linear :
  ∃! m, IsLinearFunction (f m) ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_minus_one_makes_f_linear_l799_79969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_point_l799_79980

theorem sin_double_angle_specific_point :
  ∀ α : ℝ,
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos α) = 1 ∧ r * (Real.sin α) = -2) →
  Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_point_l799_79980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l799_79911

-- Define the given values
noncomputable def presentValue : ℝ := 199.99999999999997
def futureValue : ℝ := 242
def years : ℕ := 2

-- Define the interest rate we want to prove
noncomputable def targetRate : ℝ := 0.1

-- Define the compound interest formula
noncomputable def compoundInterestFormula (r : ℝ) : ℝ := 
  futureValue / ((1 + r) ^ years)

-- Theorem statement
theorem interest_rate_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  abs (compoundInterestFormula targetRate - presentValue) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l799_79911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_2x_l799_79965

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

/-- The resulting function after transformations -/
noncomputable def h (x : ℝ) : ℝ := Real.sin (x + 2 * Real.pi / 3)

/-- Theorem stating the equivalence of the transformed function -/
theorem transform_sin_2x (x : ℝ) :
  h (2 * x) = f (x + Real.pi / 3) := by
  -- Expand the definitions of h and f
  unfold h f
  -- Simplify the expressions
  simp [Real.sin_add]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_2x_l799_79965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_climbing_ways_l799_79977

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The conjugate of the golden ratio -/
noncomputable def φ_bar : ℝ := (1 - Real.sqrt 5) / 2

/-- The number of ways to climb n stairs -/
noncomputable def F (n : ℕ) : ℝ := (1 / Real.sqrt 5) * (φ^(n+1) - φ_bar^(n+1))

theorem stair_climbing_ways (n : ℕ) :
  F n = (1 / Real.sqrt 5) * (φ^(n+1) - φ_bar^(n+1)) ∧
  φ^2 = φ + 1 ∧
  φ_bar^2 = φ_bar + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_climbing_ways_l799_79977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l799_79907

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, 
    (x₁^2 + 6*x₁ + 6*x₁*(Real.sqrt (x₁ + 2)) = 24) ∧ 
    (x₂^2 + 6*x₂ + 6*x₂*(Real.sqrt (x₂ + 2)) = 24) ∧ 
    (abs (x₁ - 20.105) < 0.001) ∧ 
    (abs (x₂ - 0.895) < 0.001) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l799_79907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_combination_difference_l799_79954

def A (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

def C (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem permutation_combination_difference : A 6 3 - C 6 3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_combination_difference_l799_79954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_squared_l799_79966

/-- An ellipse with foci and a point satisfying certain conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  A : ℝ × ℝ
  h₁ : a > b
  h₂ : b > 0
  h₃ : P.1^2 / a^2 + P.2^2 / b^2 = 1  -- P lies on the ellipse
  h₄ : (P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.2 - F₂.2) * (F₂.2 - F₁.2) = 0  -- PF₂ ⊥ F₁F₂
  h₅ : A.2 = 0  -- A lies on x-axis
  h₆ : (P.1 - A.1) * (P.1 - F₁.1) + (P.2 - A.2) * (P.2 - F₁.2) = 0  -- PA ⊥ F₁P
  h₇ : (A.1 - F₂.1)^2 + (A.2 - F₂.2)^2 = (c/2)^2  -- |AF₂| = c/2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Theorem stating that the square of the eccentricity equals (3-√5)/2 -/
theorem eccentricity_squared (e : Ellipse) : (eccentricity e)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_squared_l799_79966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_overtime_eligibility_l799_79975

/-- Calculates the number of hours after which an employee is eligible for overtime -/
def overtime_eligibility (regular_wage : ℚ) (total_hours : ℕ) (days_worked : ℕ) (total_earnings : ℚ) : ℕ :=
  let overtime_wage := regular_wage * 3 / 2
  let x := (total_earnings - overtime_wage * total_hours) / (regular_wage - overtime_wage) / days_worked
  (Int.ceil x).toNat

/-- Theorem stating that given Tina's work conditions, she is eligible for overtime after 4 hours -/
theorem tina_overtime_eligibility :
  overtime_eligibility 18 50 5 990 = 4 := by
  sorry

#eval overtime_eligibility 18 50 5 990

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_overtime_eligibility_l799_79975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_neg_one_l799_79968

/-- The function f(x) = x(e^x + ae^(-x)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

/-- f is an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- Theorem: If f(x) = x(e^x + ae^(-x)) is an even function for x ∈ ℝ, then a = -1 -/
theorem even_function_implies_a_eq_neg_one (a : ℝ) : 
  is_even (f a) → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_neg_one_l799_79968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l799_79989

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

theorem equal_area_division (ABC : Triangle) 
  (D E F G : Point) 
  (h1 : distance ABC.A ABC.B = 180)
  (h2 : distance ABC.A ABC.C = 204)
  (h3 : ∃ (k : ℝ), triangleArea ⟨ABC.A, D, ABC.C⟩ = k ∧
                   triangleArea ⟨D, E, ABC.C⟩ = k ∧
                   triangleArea ⟨E, F, ABC.C⟩ = k ∧
                   triangleArea ⟨F, G, ABC.C⟩ = k ∧
                   triangleArea ⟨G, ABC.B, ABC.C⟩ = k) :
  distance ABC.A F + distance ABC.A G = 172.5 := by
  sorry

#check equal_area_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l799_79989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l799_79921

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.sin x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 (2 * Real.pi) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ f c) ∧
  f c = Real.pi := by
  sorry

#check max_value_of_f_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l799_79921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_height_l799_79987

/-- Given a cube with edge length 2 cm and a point light source y cm directly above an upper vertex,
    if the area of the shadow (excluding the area beneath the cube) is 112 sq cm,
    then the greatest integer not exceeding 1000y is 8770. -/
theorem shadow_height (y : ℝ) : 
  (2 : ℝ) > 0 ∧ y > 0 ∧ 
  (Real.sqrt ((112 : ℝ) + 2^2) - 2)^2 / 2^2 * 2 = y →
  ⌊1000 * y⌋ = 8770 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_height_l799_79987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l799_79962

-- Define the hyperbola
noncomputable def hyperbola (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / k = 1

-- Define the eccentricity
noncomputable def eccentricity (k : ℝ) : ℝ :=
  Real.sqrt (1 + k / 4)

-- Theorem statement
theorem hyperbola_k_range (k : ℝ) :
  (∀ x y, hyperbola k x y) →
  (1 < eccentricity k ∧ eccentricity k < 2) →
  0 < k ∧ k < 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l799_79962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l799_79951

/-- The sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ → ℝ
  | 0, t => 2 * t - 2  -- Added case for n = 0
  | 1, t => 2 * t - 2
  | (n + 2), t => 2 * (t^(n+2) - 1) * a (n+1) t / (a (n+1) t + 2 * t^(n+1) - 2)

/-- Properties of the sequence a_n -/
theorem sequence_properties (t : ℝ) (ht : t ≠ 1 ∧ t ≠ -1) :
  (∀ n : ℕ, n > 0 → a n t = 2 * (t^n - 1) / n) ∧
  (t > 0 → ∀ n : ℕ, n > 0 → a (n + 1) t > a n t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l799_79951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l799_79952

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem statement
def triangle_problem (t : Triangle) (m n : Real × Real) : Prop :=
  -- Given conditions
  m.1 = Real.cos (t.A - t.B) ∧
  m.2 = Real.sin (t.A - t.B) ∧
  n.1 = Real.cos t.B ∧
  n.2 = -Real.sin t.B ∧
  m.1 * n.1 + m.2 * n.2 = -3/5 ∧
  t.a = 4 * Real.sqrt 2 ∧
  t.b = 5 ∧
  -- Conclusions to prove
  Real.sin t.A = 4/5 ∧
  t.B = Real.pi / 4 ∧
  -(t.c * Real.cos t.B) = -Real.sqrt 2 / 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (m n : Real × Real) :
  triangle_problem t m n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l799_79952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l799_79974

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/3) * Real.cos x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (Real.pi/6) (Real.pi/3) ∧
  f x = 0 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (Real.pi/6) (Real.pi/3) → f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l799_79974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colony_ratio_rounded_l799_79908

theorem colony_ratio_rounded (total_colonies : ℕ) (agreeing_colonies : ℕ)
  (h1 : total_colonies = 16)
  (h2 : agreeing_colonies = 8)
  : (agreeing_colonies : ℚ) / total_colonies = 1/2 := by
  rw [h1, h2]
  norm_num

#eval (8 : ℚ) / 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colony_ratio_rounded_l799_79908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_determination_l799_79927

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℚ  -- Principal amount
  rate : ℚ       -- Interest rate (as a percentage)
  years : ℚ      -- Loan duration in years
  interest : ℚ   -- Total interest paid

/-- Calculates the total interest for a simple interest loan -/
def calculateInterest (loan : SimpleLoan) : ℚ :=
  loan.principal * loan.rate * loan.years / 100

theorem interest_rate_determination (loan : SimpleLoan) 
  (h1 : loan.principal = 1200)
  (h2 : loan.rate = loan.years)
  (h3 : loan.interest = 432)
  (h4 : loan.interest = calculateInterest loan) :
  loan.rate = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_determination_l799_79927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_properties_l799_79999

-- Define the tetrahedron volume function
noncomputable def F : ℝ → ℝ := sorry

-- Define the domain of F
def domain : Set ℝ := { x : ℝ | 0 < x ∧ x < Real.sqrt 3 }

-- State the theorem
theorem tetrahedron_volume_properties :
  ∃ (x_max : ℝ), x_max ∈ domain ∧
  (∀ x ∈ domain, F x ≤ F x_max) ∧
  (∃ x y, x ∈ domain ∧ y ∈ domain ∧ x < y ∧ F x > F y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_properties_l799_79999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l799_79967

/-- The area of a parallelogram with base 11 and altitude twice the base is 242 -/
theorem parallelogram_area (base altitude : ℝ) 
  (h1 : base = 11) 
  (h2 : altitude = 2 * base) : 
  base * altitude = 242 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l799_79967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_parabola_l799_79934

/-- The equation |y-3| = √((x+4)² + y²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
    ∀ (x y : ℝ), (|y - 3| = Real.sqrt ((x + 4)^2 + y^2)) ↔ 
    (y = a * x^2 + b * x + c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_parabola_l799_79934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_superhero_distance_comparison_l799_79933

/-- Superhero's speed in miles per hour -/
noncomputable def superhero_speed : ℝ := 50 / (12 / 60)

/-- Supervillain's speed in miles per hour -/
def supervillain_speed : ℝ := 150

/-- Antihero's speed in miles per hour -/
def antihero_speed : ℝ := 180

/-- The difference in miles between the superhero's distance and the combined distance of the supervillain and antihero in one hour -/
noncomputable def distance_difference : ℝ := (supervillain_speed + antihero_speed) - superhero_speed

theorem superhero_distance_comparison :
  distance_difference = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_superhero_distance_comparison_l799_79933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_operations_l799_79992

noncomputable def cycle (x : ℝ) : ℝ := 1 / (3 * x^2)

noncomputable def final_result (x : ℝ) (n : ℕ) : ℝ :=
  if n % 2 = 0 then
    3 * x^(4^n)
  else
    1 / (3 * x^(4^n))

theorem calculator_operations (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  (cycle^[n] x) = final_result x n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_operations_l799_79992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_base_side_length_l799_79918

/-- Right pyramid with square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ

/-- Calculate the area of one lateral face of a right pyramid -/
noncomputable def lateral_face_area (p : RightPyramid) : ℝ :=
  (1/2) * p.base_side * p.slant_height

/-- The theorem stating the side length of the square base -/
theorem square_base_side_length :
  ∃ (p : RightPyramid), 
    p.slant_height = 20 ∧ 
    lateral_face_area p = 100 ∧ 
    p.base_side = 10 :=
by
  -- Construct the right pyramid
  let p : RightPyramid := ⟨10, 20⟩
  
  -- Prove it satisfies the conditions
  have h1 : p.slant_height = 20 := rfl
  have h2 : lateral_face_area p = 100 := by
    simp [lateral_face_area]
    norm_num
  have h3 : p.base_side = 10 := rfl

  -- Show that p is the witness that satisfies all conditions
  exact ⟨p, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_base_side_length_l799_79918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l799_79981

def is_valid (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    (a = n ∨ a = n+1 ∨ a = n+2 ∨ a = n+3) ∧
    (b = n ∨ b = n+1 ∨ b = n+2 ∨ b = n+3) ∧
    (c = n ∨ c = n+1 ∨ c = n+2 ∨ c = n+3) ∧
    (d = n ∨ d = n+1 ∨ d = n+2 ∨ d = n+3) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    8 ∣ a ∧ 27 ∣ b ∧ 25 ∣ c ∧ 49 ∣ d

theorem smallest_valid_number : (∀ m < 392, ¬is_valid m) ∧ is_valid 392 := by
  sorry

#check smallest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l799_79981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_spiral_from_arc_length_property_l799_79945

/-- A curve in polar coordinates (r, φ) with the property that the arc length between
    any two points is proportional to the difference in their distances from the origin
    is a logarithmic spiral. -/
theorem logarithmic_spiral_from_arc_length_property
  (r : ℝ → ℝ) -- r is a function of φ
  (h : ∀ φ₁ φ₂ : ℝ, ∃ k : ℝ, k > 0 ∧
    (∫ (x : ℝ) in φ₁..φ₂, Real.sqrt ((r x)^2 + (deriv r x)^2)) = k * |r φ₂ - r φ₁|) :
  ∃ C a : ℝ, ∀ φ, r φ = C * Real.exp (a * φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_spiral_from_arc_length_property_l799_79945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_rhombus_area_ratio_is_constant_l799_79928

/-- A rhombus with a specific height-to-side ratio property -/
structure SpecialRhombus where
  side : ℝ
  height : ℝ
  height_divides_side : height * 4 = side * Real.sqrt 15

/-- The fraction of area of the inscribed circle to the area of the rhombus -/
noncomputable def circle_to_rhombus_area_ratio (r : SpecialRhombus) : ℝ :=
  (Real.pi * r.height^2) / (4 * r.side * r.height)

/-- Theorem: The circle-to-rhombus area ratio is π√15/16 -/
theorem circle_to_rhombus_area_ratio_is_constant :
  ∀ r : SpecialRhombus, circle_to_rhombus_area_ratio r = Real.pi * Real.sqrt 15 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_rhombus_area_ratio_is_constant_l799_79928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l799_79953

theorem trig_inequality : 
  ∀ x : ℝ, 0 < x ∧ x < π / 6 →
  (∀ y z : ℝ, 0 < y ∧ y < z ∧ z < π / 6 → Real.sin y < Real.sin z) →
  (∀ y z : ℝ, 0 < y ∧ y < z ∧ z < π / 6 → Real.cos z < Real.cos y) →
  Real.cos (1/2 : ℝ) > Real.tan (1/2 : ℝ) ∧ Real.tan (1/2 : ℝ) > Real.sin (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l799_79953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_transformed_functions_l799_79961

-- Define the functions h and j
noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

-- Define the intersection points of h and j
axiom intersection1 : h 1 = j 1 ∧ h 1 = 1
axiom intersection2 : h 3 = j 3 ∧ h 3 = 4
axiom intersection3 : h 5 = j 5 ∧ h 5 = 9
axiom intersection4 : h 7 = j 7 ∧ h 7 = 12

-- Define the new functions
noncomputable def h' (x : ℝ) : ℝ := h (3 * x + 2)
noncomputable def j' (x : ℝ) : ℝ := 3 * j (x + 1)

-- Theorem to prove
theorem intersection_of_transformed_functions :
  ∃ x y : ℝ, x = 2 ∧ y = 12 ∧ h' x = j' x ∧ h' x = y := by
  sorry

#check intersection_of_transformed_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_transformed_functions_l799_79961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_square_relation_l799_79931

theorem unique_sum_square_relation :
  ∃! (n p : ℕ), n > 0 ∧ p > 0 ∧ Nat.Prime p ∧
  (Finset.sum (Finset.range (n + 1)) id = 3 * Finset.sum (Finset.range (p + 1)) (λ i => i^2)) ∧
  n = 5 ∧ p = 2 := by
  sorry

#check unique_sum_square_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_square_relation_l799_79931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_ratio_cube_octahedron_l799_79993

/-- The inradius of a cube with equilateral triangle faces of side length a -/
noncomputable def cube_inradius (a : ℝ) : ℝ := a / 2

/-- The inradius of a regular octahedron with equilateral triangle faces of side length a -/
noncomputable def octahedron_inradius (a : ℝ) : ℝ := a / Real.sqrt 6

/-- The ratio of the inradius of a cube to the inradius of a regular octahedron,
    both with equilateral triangle faces of side length a, is 2/3 -/
theorem inradius_ratio_cube_octahedron (a : ℝ) (h : a > 0) :
  cube_inradius a / octahedron_inradius a = 2 / 3 := by
  sorry

#check inradius_ratio_cube_octahedron

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_ratio_cube_octahedron_l799_79993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l799_79950

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- State the theorem
theorem intersection_range (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) →
  b > -4/3 ∧ b < 28/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l799_79950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_logs_l799_79920

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + |x| + (Real.cos x) / x

-- State the theorem
theorem sum_of_f_logs : 
  f (Real.log 2) + f (Real.log (1/2)) + f (Real.log 5) + f (Real.log (1/5)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_logs_l799_79920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_intersection_angle_l799_79998

theorem train_intersection_angle (θ α β : ℝ) 
  (h_acute : 0 < θ ∧ θ < π/2)
  (h_alpha_beta : 0 < β ∧ β < α ∧ α < π/2) :
  Real.tan θ = (2 * Real.sin α * Real.sin β) / Real.sin (α - β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_intersection_angle_l799_79998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l799_79939

theorem root_of_equation (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 6 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l799_79939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l799_79916

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 3*x ≤ 0}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, Real.log (1 - x) = y}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Iic 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l799_79916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_sets_l799_79938

-- Define a function to calculate the sum of consecutive integers
def sumConsecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

-- Define a function to check if a set of consecutive integers is valid
def isValidSet (a n : ℕ) : Prop :=
  a > 2 ∧ n ≥ 2 ∧ sumConsecutive a n = 18

-- Theorem statement
theorem exactly_two_valid_sets :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 2 ∧ 
  ∀ (a n : ℕ), (a, n) ∈ s ↔ isValidSet a n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_sets_l799_79938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l799_79955

/-- The distance from a point in polar coordinates to a line given in polar form --/
noncomputable def distance_point_to_line (ρ_A : ℝ) (θ_A : ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  sorry

/-- The polar equation of the line --/
noncomputable def line_equation (ρ θ : ℝ) : ℝ :=
  ρ * Real.sin (θ + Real.pi/4) - Real.sqrt 2 / 2

/-- Theorem stating the distance from the given point to the line --/
theorem distance_to_line :
  distance_point_to_line 4 (7*Real.pi/4) line_equation = Real.sqrt 2 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l799_79955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_65_l799_79976

/-- Represents a trapezoid with a perpendicular line segment inside -/
structure TrapezoidWithPerpendicularLine where
  area : ℝ
  ef_length : ℝ
  mn_length : ℝ

/-- Calculates the area of the shaded region in the trapezoid -/
noncomputable def shaded_area (t : TrapezoidWithPerpendicularLine) : ℝ :=
  t.area - 2 * (t.ef_length * t.mn_length / 2)

/-- Theorem stating that for a trapezoid with given measurements, the shaded area is 65 -/
theorem shaded_area_is_65 (t : TrapezoidWithPerpendicularLine) 
  (h_area : t.area = 117)
  (h_ef : t.ef_length = 13)
  (h_mn : t.mn_length = 4) : 
  shaded_area t = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_65_l799_79976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l799_79925

/-- Represents the speed of the train in kilometers per hour -/
noncomputable def train_speed : ℝ := 36

/-- Represents the length of the train in meters -/
noncomputable def train_length : ℝ := 50

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

/-- Calculates the time (in seconds) it takes for the train to pass a stationary point -/
noncomputable def time_to_pass (speed : ℝ) (length : ℝ) : ℝ := length / (kmph_to_mps speed)

theorem train_passing_time :
  time_to_pass train_speed train_length = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l799_79925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_l799_79990

/-- The perimeter of a rhombus with diagonals of lengths 5 and 12 is 26 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 5) (h2 : d2 = 12) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 26 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_l799_79990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l799_79946

/-- Two parabolas in a 2D plane -/
structure Parabolas where
  p1 : ℝ → ℝ
  p2 : ℝ → ℝ

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem stating the existence of an equidistant point from all intersections of two specific parabolas -/
theorem equidistant_point_exists (p : Parabolas) 
  (h1 : p.p1 = fun x ↦ -3 * x^2 + 2)
  (h2 : p.p2 = fun y ↦ -4 * y^2 + 2) :
  ∃ (ax ay : ℝ), ∀ (x y : ℝ), 
    (y = p.p1 x ∧ x = p.p2 y) → 
    distance ax ay x y = Real.sqrt (199/48) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l799_79946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l799_79936

noncomputable def scores (x : ℝ) : List ℝ := [10, x, 10, 8]

noncomputable def average (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let μ := average l
  (l.map (λ y => (y - μ)^2)).sum / l.length

theorem variance_of_scores (x : ℝ) (h : average (scores x) = 9) :
  variance (scores x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l799_79936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l799_79919

/-- The sum of the infinite series ∑(n=1 to ∞) (n^5 + 2n^3 + 5n^2 + 20n + 20) / (2^(n+1) * (n^5 + 5)) is equal to 1/2. -/
theorem infinite_series_sum : 
  ∑' n : ℕ, ((n+1)^5 + 2*(n+1)^3 + 5*(n+1)^2 + 20*(n+1) + 20) / (2^(n+2) * ((n+1)^5 + 5)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l799_79919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_pencils_theorem_l799_79902

-- Define a structure for colored pencils
structure ColoredPencil where
  color : Nat

-- Define the proposition that all pencils in a set have the same color
def all_same_color (n : ℕ) : Prop :=
  ∀ (S : Finset ColoredPencil), S.card = n → 
    (∀ x y : ColoredPencil, x ∈ S → y ∈ S → x.color = y.color)

-- State the theorem
theorem colored_pencils_theorem : 
  ∀ n : ℕ, n > 0 → all_same_color n := by
  sorry

-- Example to show the flaw in the induction
example : ¬ (∀ n : ℕ, n > 0 → all_same_color n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_pencils_theorem_l799_79902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l799_79947

/-- Represents a right cone with variable height and fixed base circumference -/
structure Cone where
  baseCircumference : ℝ
  height : ℝ

/-- Calculates the volume of a cone given its base radius and height -/
noncomputable def coneVolume (radius : ℝ) (height : ℝ) : ℝ :=
  (1/3) * Real.pi * radius^2 * height

theorem cone_height_ratio (c : Cone) (newVolume : ℝ) :
  c.baseCircumference = 12 * Real.pi →
  c.height = 20 →
  newVolume = 108 * Real.pi →
  ∃ (newHeight : ℝ),
    coneVolume (c.baseCircumference / (2 * Real.pi)) newHeight = newVolume ∧
    newHeight / c.height = 9 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l799_79947
